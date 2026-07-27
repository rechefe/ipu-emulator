"""Prove a 9-cyc/ch schedule exists for conv_bn via a 4-slot +128 cyclic ring.

Model: 4 physical 128B slots (offsets 0,128,256,384 mod 512). Each iteration is a
NEW input channel needing 3 fresh chunks (kr=-1, kr=0, kr=+1). lr_read advances
+128 per iteration. We want to issue all 3 fresh loads for iter N+1 *inside* iter
N's 9-tap window (taps 1..9), each into a slot that is not being read by any
remaining tap of iter N at the moment of the load -- AND visible (snapshot) to the
first tap of iter N that reads it... but iter N's chunks are loaded during iter N-1,
so for iter N+1 the loads happen during iter N and must be visible by iter N+1 tap 1.

Snapshot rule: a load issued in tap word K is visible starting tap word K+1.
So a chunk loaded at iter N tap K (K in 1..9) is visible from iter N tap K+1 onward,
and certainly by iter N+1 tap 1 (next iteration). The ONLY illegal case is loading a
slot in the SAME word a tap reads it, or clobbering a slot iter N still needs to read.

Per iter, which slots are read at which tap (kr=-1 = read base R-128, kr=0 = R, kr=+1 = R+128):
  taps 1,2,3 -> kr=-1 slot (R-128)
  taps 4,5,6 -> kr=0  slot (R)
  taps 7,8,9 -> kr=+1 slot (R+128)
(Each tap reads a byte-window within its slot; for slot-occupancy we treat the whole
128B slot as live for taps that target it. The walking pointer crosses slot boundaries
but the kr grouping above is the dominant occupancy.)
"""

SLOTS = [0, 128, 256, 384]


def slot_of(byte_off):
    return (byte_off % 512) // 128


def simulate():
    # lr_read for iter 0; advance +128 each iter.
    # We need, for each iter, the 3 chunk slots and the last tap each is read.
    # last-read tap: kr=-1 -> tap 3, kr=0 -> tap 6, kr=+1 -> tap 9.
    last_read = {"-1": 3, "0": 6, "+1": 9}

    # For iter N+1's loads we have iter N's 9 taps as load opportunities.
    # iter N+1 needs slots:
    #   kr=-1 -> (R+128 -128) = R          (iter N's kr=0 slot)
    #   kr=0  -> (R+128)                    (iter N's kr=+1 slot)
    #   kr=+1 -> (R+128 +128) = R+256       (iter N's kr=-1 slot, since R-128 == R+384-512... check)
    # iter N slots are {R-128, R, R+128}. R+256 mod 512 vs R-128 mod 512:
    #   R+256 != R-128 (differ by 384). So R+256 is the 4TH slot (the free one).
    # Good: the +128 ring means iter N+1's kr=+1 lands in the slot iter N never used.
    R = 0
    print("Per-iteration slot occupancy (offsets mod 512):")
    for N in range(6):
        km1 = (R - 128) % 512
        k0 = R % 512
        kp1 = (R + 128) % 512
        free = (R + 256) % 512  # the 4th slot this iter doesn't read
        print(f"  iter {N}: lr_read={R:3d}  kr=-1@{km1:3d} kr=0@{k0:3d} kr=+1@{kp1:3d}  (free 4th slot @{free:3d})")
        R = (R + 128) % 512

    print()
    print("iter N+1 fresh-load targets (where each new chunk must land), vs iter N read-windows:")
    R = 0
    for N in range(5):
        # iter N's read windows by tap
        nN = {"-1": (R - 128) % 512, "0": R % 512, "+1": (R + 128) % 512}
        Rn = (R + 128) % 512  # iter N+1 lr_read
        nN1 = {"-1": (Rn - 128) % 512, "0": Rn % 512, "+1": (Rn + 128) % 512}
        # For each iter N+1 chunk, find which iter-N slot it reuses and the earliest
        # iter-N tap at which that slot is no longer read (free to overwrite).
        print(f"  -- loading for iter {N+1} during iter {N} (lr_read N={R}, N+1={Rn}) --")
        for kr in ("-1", "0", "+1"):
            target = nN1[kr]
            # which of iter N's slots is this? find matching read-window, or it's the free 4th
            owner = None
            for k2, off in nN.items():
                if off == target:
                    owner = k2
            if owner is None:
                earliest_free_tap = 1  # iter N never reads it -> free from tap 1
                why = "iter N's free 4th slot"
            else:
                earliest_free_tap = last_read[owner] + 1  # free the tap AFTER its last read
                why = f"iter N kr={owner} slot, last read tap {last_read[owner]}"
            print(f"     iter{N+1} kr={kr:>2} -> slot @{target:3d}: free from tap {earliest_free_tap}  ({why})")
        R = Rn
        print()


def search_orders():
    """Accumulation is commutative -> the 9 taps may be visited in any order.
    For a given visit order, each kr-group's 'last read tap' is the max position
    (1..9) at which any of its 3 taps appears. iter N+1's three fresh chunks need:
       kr=-1 -> slot = iterN kr=0   (free after iterN kr=0  group's last tap)
       kr= 0 -> slot = iterN kr=+1  (free after iterN kr=+1 group's last tap)
       kr=+1 -> slot = iterN free4th (free from tap 1 always)
    A load issued at tap K is visible from K+1; it must be issued at a tap in
    [free_tap .. 9] (within the window) for the chunk to be ready by next iter tap 1.
    We need a free XMEM slot at that tap too, but XMEM has 1 slot/word and at most
    a couple taps already use it -> check feasibility = each needed load has
    free_tap <= 9 AND we can assign distinct tap words.
    """
    from itertools import permutations

    groups = {"-1": None, "0": None, "+1": None}
    # positions 1..9; assign 3 consecutive-or-not positions per group.
    # Brute force over which 3 positions each group occupies is huge; instead
    # parametrize by the ORDER of the 9 (kr,kc) taps. We only care about each
    # group's max position.
    krs = ["-1", "-1", "-1", "0", "0", "0", "+1", "+1", "+1"]
    best = []
    seen = set()
    for perm in set(permutations(krs)):
        # last position (1-indexed) of each group
        lastpos = {}
        for i, kr in enumerate(perm, start=1):
            lastpos[kr] = i
        # free taps for the 3 loads (load may issue the tap AFTER last read):
        free_km1 = lastpos["0"] + 1   # iterN+1 kr=-1 needs iterN kr=0 slot
        free_k0 = lastpos["+1"] + 1   # iterN+1 kr= 0 needs iterN kr=+1 slot
        free_kp1 = 1                  # free 4th slot
        loads = sorted([free_km1, free_k0, free_kp1])
        # need 3 distinct tap-words in [free .. 9]; greedily assignable iff the
        # i-th smallest free-tap <= 9-(2-i)? Simpler: all <=9 and we can pick
        # distinct words >= each free tap within 1..9. Feasible iff sorted free
        # taps f1<=f2<=f3 satisfy f1<=7, f2<=8, f3<=9 (need room for 3 distinct).
        f1, f2, f3 = loads
        feasible = (f1 <= 7) and (f2 <= 8) and (f3 <= 9)
        key = (lastpos["-1"], lastpos["0"], lastpos["+1"])
        if feasible and key not in seen:
            seen.add(key)
            best.append((key, free_km1, free_k0, free_kp1, perm))
    print(f"\nFeasible group-last-position signatures (kr-1,kr0,kr+1 last tap): {len(best)}")
    for key, fa, fb, fc, perm in best[:12]:
        print(f"  last-read (kr-1,0,+1)={key}  load-free taps: kr-1@{fa} kr0@{fb} kr+1@{fc}  e.g. order={perm}")
    if not best:
        print("  NONE -> no tap reordering yields a 9-cyc schedule with 4 slots.")


def check_contiguous_runs():
    """Practical constraint: the walking pointer is cheapest when each kr ROW is a
    contiguous run of 3 taps (kc=-1,0,+1) so within a row the walk is just +1,+1.
    Between rows it strides by (cols-2). So realistic orders are permutations of the
    THREE ROW-BLOCKS [kr=-1 block], [kr=0 block], [kr=+1 block], each block = 3 taps.
    Last-read tap of a block placed in slot p (p=1st/2nd/3rd block) is 3p.
    We need:
       free_km1 = (kr=0 block last tap) + 1  <= 9
       free_k0  = (kr=+1 block last tap) + 1 <= 9   (the hard one)
       free_kp1 = 1
    and 3 distinct load words assignable.
    """
    from itertools import permutations
    print("\nRow-block orderings (each block contiguous, 3 taps):")
    any_ok = False
    for order in permutations(["-1", "0", "+1"]):
        last = {kr: 3 * (i + 1) for i, kr in enumerate(order)}
        free_km1 = last["0"] + 1
        free_k0 = last["+1"] + 1
        free_kp1 = 1
        loads = sorted([free_km1, free_k0, free_kp1])
        f1, f2, f3 = loads
        feasible = (f1 <= 7) and (f2 <= 8) and (f3 <= 9)
        print(f"  blocks {order}: last(kr0)={last['0']} last(kr+1)={last['+1']}  "
              f"loads kr-1@{free_km1} kr0@{free_k0} kr+1@{free_kp1}  -> {'OK' if feasible else 'NO'}")
        any_ok = any_ok or feasible
    print("  => feasible with contiguous row-blocks!" if any_ok
          else "  => NO contiguous-row-block order works; need split rows.")


if __name__ == "__main__":
    simulate()
    search_orders()
    check_contiguous_runs()
