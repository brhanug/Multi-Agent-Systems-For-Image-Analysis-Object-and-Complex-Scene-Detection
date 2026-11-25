# Level 4: what the 0.6/0.4 mix in the deployed triage score does (plan, 9 Nov 2026, before running)

The deployed routing score is U_triage = 0.6 * spread + 0.4 * (1 - mean confidence) over the four deployed
signals; 0.6/0.4 was set by hand and never tested except against pure confidence at archive scale.

On the 1,250-picture pool, the deployed signals and decision (SAA at 0.5) and the same 200 book-disjoint
60/20/20 redraws (seed 20260926): U_w = w * spread + (1 - w) * (1 - mean confidence) for w = 0, 0.1, ..., 1
(w = 0.6 is the deployed score), each calibrated by conformal risk control on the calibration books.
Reported per w: approval and realised risk at alpha 0.05 and 0.10 on the test books, and the AUROC with
which U_w ranks the errors on the test books. Also a learned w: chosen on the training books (the w with
the highest error AUROC there) and scored on the test books. Reading: where 0.6 lies on the curve, and
whether choosing w from data helps.
