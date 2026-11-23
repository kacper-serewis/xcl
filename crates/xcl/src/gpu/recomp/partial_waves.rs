//! Compute groups that are not whole console waves: can the program tell the lanes its last
//! wave does not launch?
//!
//! A group of 40 threads is one wave64 with lanes 0..39 launched. On the console lanes 40..63
//! exist all the same: inactive, but `v_writelane` (which ignores EXEC) writes them, and code
//! that turns them on in EXEC – whole-wave code (`s_or_saveexec_b64 s, -1`), `s_not_b64 exec`,
//! `s_wqm_b64 exec` – runs in them: its compares see them, its writes reach them, a later
//! `v_readlane` or DPP step reads them back. A PC thread group of 40 threads has no such
//! lanes. A program that can tell gets them (`Em::cs_pad`: the group padded to whole waves,
//! the padding lanes inactive at the start, as on the console); any other program keeps its
//! group, and its translation, as they are.
//!
//! A program can tell when it has a `v_writelane` to a lane the last wave may not launch, or
//! when EXEC may get a bit outside the launched lanes. The second is a forward data-flow pass
//! over the SGPRs as 32-bit words that are known subsets of the last wave's launched lanes,
//! as a low word (lanes 0..31) and as a high word (lanes 32..63): EXEC is one at the start; a
//! VALU mask (`v_cmp*`, carries) is a subset of EXEC; AND with a subset, OR / XOR of two, the
//! saveexec forms, moves and selects keep the property; every other value is taken to have
//! any bit. Conservative: where it cannot follow (an indirect jump, a value from memory) the
//! program is padded, which is exact, only more threads.

use std::collections::{BTreeMap, BTreeSet};

use super::upper_lanes::{is_valu, sgpr_outputs, slot, successors};
use crate::gpu::rdna2::{self, Decoded};

/// What is known of a 32-bit word: `LO` – its bits are launched lanes when it is a low word,
/// `HI` – when it is a high word.
type Sub = u8;
const LO: Sub = 1;
const HI: Sub = 2;

#[derive(Clone, PartialEq)]
struct State {
    s: [Sub; 128],
    /// launched lanes of the last wave as (low, high) words
    launched: (u32, u32),
}

impl State {
    fn of(&self, v: u32) -> Sub {
        (if v & !self.launched.0 == 0 { LO } else { 0 }) | (if v & !self.launched.1 == 0 { HI } else { 0 })
    }

    /// What any value is.
    fn any(&self) -> Sub {
        self.of(!0)
    }

    fn entry(wave64: bool, launched: u32) -> State {
        let mask = if launched >= 64 { !0u64 } else { (1u64 << launched) - 1 };
        let mut st = State { s: [0; 128], launched: (mask as u32, (mask >> 32) as u32) };
        st.s = [st.any(); 128];
        st.s[125] = LO | HI; // NULL
        st.s[126] |= LO;
        if wave64 {
            st.s[127] |= HI;
        }
        st
    }

    fn join(&self, o: &State) -> State {
        let mut s = self.s;
        for (a, b) in s.iter_mut().zip(o.s) {
            *a &= b;
        }
        State { s, launched: self.launched }
    }

    fn set(&mut self, r: u64, w: Sub) {
        if r < 128 && r != 125 {
            self.s[r as usize] = w;
        }
    }

    /// A 64-bit write: to NULL (125) it writes nothing, not 126 (EXEC_LO) either.
    fn set64(&mut self, r: u64, (lo, hi): (Sub, Sub)) {
        if r != 125 {
            self.set(r, lo);
            self.set(r + 1, hi);
        }
    }

    /// A 32-bit scalar source (operand code).
    fn src32(&self, d: &Decoded, code: u64) -> Sub {
        match code {
            0..=127 => self.s[code as usize],
            128..=192 => self.of((code - 128) as u32),
            193..=208 => self.of((-((code - 192) as i32)) as u32),
            255 => d.literal.map_or(self.any(), |l| self.of(l)),
            _ => self.any(),
        }
    }

    /// A 64-bit scalar source (operand code) as (low, high) words; a 32-bit literal is the
    /// low word of an integer operand, inline integers are sign-extended (as `Em::code64`).
    fn src64(&self, d: &Decoded, code: u64) -> (Sub, Sub) {
        match code {
            0..=123 | 126 => (self.s[code as usize], self.s[code as usize + 1]),
            125 => (LO | HI, LO | HI),
            128..=192 => (self.of((code - 128) as u32), self.of(0)),
            193..=208 => (self.of((-((code - 192) as i32)) as u32), self.of(!0)),
            255 => (d.literal.map_or(self.any(), |l| self.of(l)), self.of(0)),
            _ => (self.any(), self.any()),
        }
    }

    /// A bitwise SALU operation by its name's second part (`S_ANDN2_SAVEEXEC_B64` → "ANDN2"):
    /// a AND b is a subset of either, a OR / XOR b of what both are subsets of, a AND NOT b
    /// of a; the forms that invert their result or an OR operand may set any bit.
    fn bitop(&self, op: &str, a: Sub, b: Sub) -> Option<Sub> {
        Some(match op {
            "AND" => a | b,
            "OR" | "XOR" => a & b,
            "ANDN2" => a,
            "ANDN1" => b,
            "ORN2" | "NAND" | "NOR" | "XNOR" | "ORN1" => self.any(),
            _ => return None,
        })
    }
}

/// One instruction's effect.
fn step(d: &Decoded, st: &mut State, wave64: bool) {
    let name = d.inst.name;
    let fv = |f: &str| rdna2::field_value(d.enc, &d.words, f).unwrap_or(0);
    let any = st.any();
    if name.starts_with("S_MOVREL") {
        // s_movreld / s_movrelsd_2 write s[dst + m0]: any SGPR up to VCC (`x64_masks::uses`)
        for r in 0..124 {
            st.set(r, any);
        }
        return;
    }
    let op = name.split('_').nth(1).unwrap_or("");
    let b64 = name.ends_with("_B64");
    if name.contains("SAVEEXEC") || name.contains("WREXEC") {
        // a wave32 program's _B64 forms are its _B32 ones, as `Em` translates them: EXEC_HI
        // and the pair's high word are not written
        let b64 = b64 && wave64;
        let s0 = if b64 { st.src64(d, fv("SSRC0")) } else { (st.src32(d, fv("SSRC0")), any) };
        let (el, eh) = (st.s[126], st.s[127]);
        let nl = st.bitop(op, s0.0, el).unwrap_or(any);
        let nh = st.bitop(op, s0.1, eh).unwrap_or(any);
        let dst = fv("SDST");
        let wr = name.contains("WREXEC");
        st.s[126] = nl;
        if b64 {
            st.s[127] = nh;
            st.set64(dst, if wr { (nl, nh) } else { (el, eh) });
        } else {
            st.set(dst, if wr { nl } else { el });
        }
        return;
    }
    if d.enc.name == "ENC_SOP1" || d.enc.name == "ENC_SOP2" {
        let dst = fv("SDST");
        let a = |st: &State| if b64 { st.src64(d, fv("SSRC0")) } else { (st.src32(d, fv("SSRC0")), any) };
        let b = |st: &State| if b64 { st.src64(d, fv("SSRC1")) } else { (st.src32(d, fv("SSRC1")), any) };
        let both = |x: (Sub, Sub), y: (Sub, Sub)| (x.0 & y.0, x.1 & y.1);
        let r: Option<(Sub, Sub)> = match op {
            "MOV" if d.enc.name == "ENC_SOP1" && (name == "S_MOV_B32" || name == "S_MOV_B64") => Some(a(st)),
            "CMOV" if d.enc.name == "ENC_SOP1" => Some(both(a(st), (st.s[dst.min(127) as usize], st.s[(dst + 1).min(127) as usize]))),
            "CSELECT" => Some(both(a(st), b(st))),
            _ if d.enc.name == "ENC_SOP2" && (name.ends_with("_B32") || b64) => {
                let (x, y) = (a(st), b(st));
                st.bitop(op, x.0, y.0).zip(st.bitop(op, x.1, y.1))
            }
            _ => None,
        };
        if let Some((lo, hi)) = r {
            if b64 {
                st.set64(dst, (lo, hi));
            } else {
                st.set(dst, lo);
            }
            return;
        }
    }
    let valu = is_valu(d);
    let (el, eh) = (st.s[126], st.s[127]);
    for (r, n) in sgpr_outputs(d) {
        if valu && n == 2 {
            // a lane mask (v_cmpx: EXEC itself): the bits of inactive lanes are 0. A wave32
            // program's mask is its low word only
            if wave64 {
                st.set64(r, (el, eh));
            } else {
                st.set(r, el);
            }
        } else {
            for k in r..r + n {
                st.set(k, any);
            }
        }
    }
}

/// Why the program can tell the lanes the last wave of its group does not launch (None: it
/// cannot). `insts`: the program in address order, far jumps resolved; `launched`: the lanes
/// that wave does launch (1 .. wave size - 1).
pub fn sees_lanes_not_launched(insts: &[Decoded], wave64: bool, launched: u32) -> Option<&'static str> {
    let lanes = if wave64 { 64 } else { 32 };
    if insts.is_empty() || launched == 0 || launched >= lanes {
        return None;
    }
    // `v_writelane` ignores EXEC: a lane that was not launched is written on the console
    if insts.iter().any(|d| d.inst.name == "V_WRITELANE_B32" && !slot(d).is_some_and(|(_, l)| l & (lanes - 1) < launched)) {
        return Some("v_writelane to a lane the last wave may not launch");
    }
    let pcs: BTreeMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    let succ = successors(insts, &pcs);
    let mut at: Vec<Option<State>> = vec![None; insts.len()];
    at[0] = Some(State::entry(wave64, launched));
    let mut work = BTreeSet::from([0usize]);
    while let Some(i) = work.pop_first() {
        let mut st = at[i].clone().expect("queued with a state");
        step(&insts[i], &mut st, wave64);
        if st.s[126] & LO == 0 || (wave64 && st.s[127] & HI == 0) {
            return Some("EXEC may turn on lanes the last wave does not launch");
        }
        let Some(s) = &succ[i] else {
            return Some("an indirect jump: EXEC is not followed");
        };
        for &j in s {
            let new = match &at[j] {
                None => st.clone(),
                Some(old) => old.join(&st),
            };
            if at[j].as_ref() != Some(&new) {
                at[j] = Some(new);
                work.insert(j);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;

    const V: u64 = 256;
    type I = (&'static str, &'static str, &'static [(&'static str, u64)]);
    const ENDPGM: I = ("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)]);
    const CMP: I = ("V_CMP_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]);

    fn program(lines: &[I]) -> Vec<Decoded> {
        decoded(lines.iter().flat_map(|(n, e, f)| encode(n, e, f, None).unwrap()).collect())
    }

    fn decoded(bytes: Vec<u8>) -> Vec<Decoded> {
        let mut out = Vec::new();
        let mut off = 0usize;
        while off < bytes.len() {
            let d = rdna2::decode(&bytes[off..], off as u64).unwrap();
            off += d.len;
            out.push(d);
        }
        rdna2::resolve_far_jumps(&mut out);
        out
    }

    fn sees(lines: &[I], launched: u32) -> bool {
        sees_lanes_not_launched(&program(lines), true, launched).is_some()
    }

    #[test]
    fn if_else_and_loop_masks_stay_in_the_launched_lanes() {
        // if (vcc) … else … end, as the console compiler writes it
        let if_else: &[I] = &[
            CMP,
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]),
            ("S_XOR_B64", "ENC_SOP2", &[("SDST", 0), ("SSRC0", 126), ("SSRC1", 0)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 129)]),
            ("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 0)]),
            ("S_XOR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 3), ("SRC0", 130)]),
            ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 0)]),
            ENDPGM,
        ];
        // a loop that lanes leave one by one: s[4:5] collects them, EXEC loses them
        let a_loop: &[I] = &[
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 128)]),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 6), ("SSRC0", 126)]),
            CMP,
            ("S_AND_B64", "ENC_SOP2", &[("SDST", 2), ("SSRC0", 126), ("SSRC1", 106)]),
            ("S_OR_B64", "ENC_SOP2", &[("SDST", 4), ("SSRC0", 2), ("SSRC1", 4)]),
            ("S_ANDN2_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 4)]),
            ("S_CBRANCH_EXECNZ", "ENC_SOPP", &[("SIMM16", 0xFFFB)]),
            ("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 6)]),
            ENDPGM,
        ];
        for launched in [1, 10, 20, 32, 40, 63] {
            assert!(!sees(if_else, launched), "if / else, {launched} lanes");
            assert!(!sees(a_loop, launched), "loop, {launched} lanes");
            assert!(!sees(&[("V_CMPX_GT_U32", "ENC_VOPC", &[("SRC0", V), ("VSRC1", 1)]), ENDPGM], launched), "v_cmpx, {launched} lanes");
        }
    }

    #[test]
    fn exec_turning_on_other_lanes_is_seen() {
        let widen: [&[I]; 7] = [
            &[("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 193)])],
            &[("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 193)])],
            &[("S_NOT_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)])],
            &[("S_WQM_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 126)])],
            &[("S_ORN2_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 106)])],
            // a mask from the program's arguments, and one from a scalar load
            &[("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 8)])],
            &[("S_LOAD_DWORDX2", "ENC_SMEM", &[("SDATA", 12), ("SBASE", 4), ("OFFSET", 0), ("SOFFSET", 125)]), ("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 12)])],
        ];
        for w in widen {
            let mut l = vec![CMP];
            l.extend_from_slice(w);
            l.push(ENDPGM);
            for launched in [10, 40] {
                assert!(sees(&l, launched), "{} with {launched} lanes", w.last().unwrap().0);
            }
        }
    }

    #[test]
    fn a_wave32_saveexec_b64_writes_one_word() {
        // in a wave32 program `s_and_saveexec_b64 s[4:5]` writes s4 only (as the translation
        // does): s5 keeps the program's argument, and EXEC_LO taken from it may hold any lane
        let l: &[I] = &[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 126)]),
            CMP,
            ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 4), ("SSRC0", 106)]),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 5)]),
            ENDPGM,
        ];
        assert!(sees_lanes_not_launched(&program(l), false, 20).is_some());
        // the same from s4, the word it does write, stays in the launched lanes
        let l: &[I] = &[l[0], l[1], l[2], ("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 4)]), ENDPGM];
        assert!(sees_lanes_not_launched(&program(l), false, 20).is_none());
    }

    #[test]
    fn a_saveexec_to_null_that_widens_exec_is_seen() {
        // EXEC_HI = 0, then `s_or_saveexec_b64 null, 0xFFFFFFFF`: lanes 20..31 turn on. The
        // pair's high word must not land in EXEC_LO (s126) and hide that
        let widen = [
            encode("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 128)], None),
            encode("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 125), ("SSRC0", 255)], Some(!0)),
            encode("S_ENDPGM", "ENC_SOPP", &[("SIMM16", 0)], None),
        ];
        let widen = decoded(widen.into_iter().flat_map(Result::unwrap).collect());
        assert!(sees_lanes_not_launched(&widen, true, 20).is_some());
    }

    #[test]
    fn masks_discarded_into_null_leave_exec_alone() {
        // EXEC stays inside the launched lanes
        let narrow: [&[I]; 3] = [
            &[CMP, ("S_AND_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 125), ("SSRC0", 106)])],
            &[CMP, ("S_AND_B64", "ENC_SOP2", &[("SDST", 125), ("SSRC0", 126), ("SSRC1", 106)])],
            &[("V_CMP_GT_U32", "ENC_VOP3", &[("VDST", 125), ("SRC0", V), ("SRC1", V + 1)])],
        ];
        let mut seen = Vec::new();
        for n in narrow {
            let mut l = n.to_vec();
            l.push(ENDPGM);
            for launched in [20, 40] {
                if sees(&l, launched) {
                    seen.push(format!("{} with {launched} lanes", n.last().unwrap().0));
                }
            }
        }
        assert!(seen.is_empty(), "{seen:?}");
    }

    #[test]
    fn constants_count_by_their_bits() {
        // exec = 15 (lanes 0..3): inside 10 or 4 launched lanes, not inside 3
        let low8: &[I] = &[("S_MOV_B64", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 128 + 15)]), ENDPGM];
        assert!(!sees(low8, 10) && !sees(low8, 4) && sees(low8, 3));
        // the high word alone: 1 = lane 32, launched from 33 lanes on
        let hi: &[I] = &[("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 129)]), ENDPGM];
        assert!(sees(hi, 32) && !sees(hi, 33));
        // the low word of a wave of 40 lanes is whole: any value is inside it
        let lo: &[I] = &[("S_MOV_B32", "ENC_SOP1", &[("SDST", 126), ("SSRC0", 8)]), ENDPGM];
        assert!(!sees(lo, 40) && sees(lo, 31));
    }

    #[test]
    fn a_wider_exec_on_one_path_is_seen() {
        // s_cbranch_scc0 → skip; s_mov_b64 s[2:3], -1; skip: s_and_b64 exec, exec, s[2:3] is
        // fine either way, s_or_b64 is not
        let with = |last: I| -> Vec<I> {
            vec![
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 128)]),
                ("S_CBRANCH_SCC0", "ENC_SOPP", &[("SIMM16", 1)]),
                ("S_MOV_B64", "ENC_SOP1", &[("SDST", 2), ("SSRC0", 193)]),
                last,
                ENDPGM,
            ]
        };
        assert!(!sees(&with(("S_AND_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)])), 40));
        assert!(sees(&with(("S_OR_B64", "ENC_SOP2", &[("SDST", 126), ("SSRC0", 126), ("SSRC1", 2)])), 40));
    }

    #[test]
    fn v_writelane_to_a_lane_that_may_be_missing_is_seen() {
        let lane5: &[I] = &[("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 4), ("SRC1", 128 + 5)]), ENDPGM];
        let lane20: &[I] = &[("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 4), ("SRC1", 128 + 20)]), ENDPGM];
        let lane37: &[I] = &[("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 4), ("SRC1", 128 + 37)]), ENDPGM];
        assert!(!sees(lane5, 20) && sees(lane20, 20) && sees(lane37, 20));
        // a wave32 program's lane 37 is lane 5
        assert!(sees_lanes_not_launched(&program(lane37), false, 20).is_none());
        // a lane from an SGPR: any lane
        assert!(sees(&[("V_WRITELANE_B32", "ENC_VOP3", &[("VDST", 9), ("SRC0", 4), ("SRC1", 6)]), ENDPGM], 20));
    }

    #[test]
    fn whole_waves_and_wave32_programs() {
        let whole_wave: &[I] = &[("S_OR_SAVEEXEC_B64", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 193)]), ENDPGM];
        assert!(sees_lanes_not_launched(&program(whole_wave), true, 0).is_none(), "no partial wave");
        // a wave32 program: EXEC is s126 alone, s127 is data
        let w32: &[I] = &[("S_MOV_B32", "ENC_SOP1", &[("SDST", 127), ("SSRC0", 193)]), CMP, ("S_AND_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 106)]), ENDPGM];
        assert!(sees_lanes_not_launched(&program(w32), false, 20).is_none());
        let w32_wide: &[I] = &[("S_OR_SAVEEXEC_B32", "ENC_SOP1", &[("SDST", 0), ("SSRC0", 193)]), ENDPGM];
        assert!(sees_lanes_not_launched(&program(w32_wide), false, 20).is_some());
    }
}
