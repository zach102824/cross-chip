# full CDR details for LiH testing case (2nd version)

Source: `test_LiH_case/lih_fig13_compiled_ansatz.ipynb` per-Pauli CDR cell.

## Configuration used

- bond_length: `1.5`
- target params [theta1, theta2, theta3]: `[-0.128705, 0.233859, 0.114671]`
- measurement_scheme: `ogm`
- num_shots: `8192`
- sampling_seed: `1234`
- apply_readout_noise: `True`
- apply_rem in CDR training/target estimation: `True`
- readout p_0_success: `[0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514]`
- readout p_1_success: `[0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514]`
- base_noise_cfg: `{'two_qubit_depol_prob': 0.018, 'one_qubit_depol_prob': 0.0018}`
- cdr_cfg: `{"num_circuits": 10, "t_max": 2, "seed": 42, "cdr_fit_scope": "per_pauli"}`
- per-term records included in this document: first `20` terms
- training resolvers count: `10`
- training non-Clifford counts (t_remaining): `[2, 2, 2, 2, 2, 2, 2, 2, 2, 2]`
- OGM file: `/Users/zacharyhe/shadowgrouping/haozhaowu/LiH/hamil_class/ogm_outputs/OGM_ogm_LiH_1.5.txt`

## Important note on epsilon

- For OGM `SettingSampler`, shot allocation is sampled from file probabilities `p` and total `N_samples` only; epsilon is not used in the sampler math.

## Model definition used

- Per-pauli CDR applies, for each term `k`:
  - unmit branch: `y_k ~= a_u[k] * x_u[k] + b_u[k]`
  - rem branch:   `y_k ~= a_r[k] * x_r[k] + b_r[k]`
- Total corrected energies:
  - `E_cdr_unmit = offset + sum_k w_k * (a_u[k] * x_u_target[k] + b_u[k])`
  - `E_cdr_rem   = offset + sum_k w_k * (a_r[k] * x_r_target[k] + b_r[k])`

## Fit quality diagnostics (REM per-term fits)

- LiH terms total: `117`
- Panel a terms (excl. a≈b≈0): `49` (dropped `68`)
- Panel a slope (a): mean=`1.009430`, std=`0.753895`
- All terms slope (a): mean=`0.422753`, std=`0.697172`
- Intercept (b): mean=`0.010405`, std=`0.095189`
- R^2: mean=`0.313984`, std=`0.370430`
- Terms with slope < 1: panel-a `20/49`, all `88/117`
- Terms with R^2 < 0.9: `96/117`
- Terms flagged as likely bad fits (R^2<0.9 and at least one of near-zero (a,b), weak signal, weak x spread, intercept-dominated): `82/117`

### Representative bad-fitting terms (lowest R^2)

- rank 1, term 30: R^2=`0.000001`, a=`-0.000000`, b=`0.000000`, |w_k|=`4.163813e-05`, x_std=`0.074085`, y_std=`4.463769e-09`, reasons: `near-zero-(a,b), weak-y-signal, intercept-dominated`
- rank 2, term 19: R^2=`0.000024`, a=`0.000000`, b=`-0.000000`, |w_k|=`4.479075e-03`, x_std=`0.036546`, y_std=`9.651510e-10`, reasons: `near-zero-(a,b), weak-y-signal, intercept-dominated`
- rank 3, term 66: R^2=`0.000123`, a=`0.000000`, b=`0.000000`, |w_k|=`6.575745e-03`, x_std=`0.199597`, y_std=`2.089657e-08`, reasons: `near-zero-(a,b), weak-y-signal, intercept-dominated`
- rank 4, term 49: R^2=`0.000271`, a=`0.000000`, b=`0.000000`, |w_k|=`3.305873e-02`, x_std=`0.046458`, y_std=`9.089939e-09`, reasons: `near-zero-(a,b), weak-y-signal, intercept-dominated`
- rank 5, term 68: R^2=`0.000496`, a=`-0.000000`, b=`0.000000`, |w_k|=`2.111138e-03`, x_std=`0.091218`, y_std=`1.162120e-08`, reasons: `near-zero-(a,b), weak-y-signal, intercept-dominated`
- rank 6, term 82: R^2=`0.000868`, a=`0.000000`, b=`0.000000`, |w_k|=`4.479075e-03`, x_std=`0.049717`, y_std=`9.011289e-09`, reasons: `near-zero-(a,b), weak-y-signal, intercept-dominated`
- rank 7, term 76: R^2=`0.001084`, a=`-0.026965`, b=`0.023844`, |w_k|=`1.521710e-02`, x_std=`0.035845`, y_std=`3.200427e-02`, reasons: `intercept-dominated`
- rank 8, term 92: R^2=`0.002150`, a=`0.000000`, b=`0.000000`, |w_k|=`2.775746e-03`, x_std=`0.166469`, y_std=`2.609003e-08`, reasons: `near-zero-(a,b), weak-y-signal, intercept-dominated`

### Why slope can be far below 1

- In this per-term CDR setup, each Pauli term is fit independently, so slope is not a global noise attenuation factor and can vary widely across terms.
- Many terms have near-zero exact expectation over training circuits; for those weak-signal terms, linear regression can return tiny or unstable slopes (often much smaller than 1).
- Including all terms therefore pulls down the all-term slope mean; panel-a removes near-zero coefficient pairs and better reflects active terms.

### What cases give bad fitting (R^2 not close to 1)

- Weak target signal (`std(y_exact)` tiny): shot noise and finite-sampling fluctuations dominate the regression residuals.
- Weak feature spread (`std(x_rem)` tiny): the fitted slope is poorly constrained by training data.
- Intercept-dominated fits (`|b| > |a|*std(x_rem)`): correction behaves like a constant offset rather than a strong linear relation.
- Near-zero `(a,b)` pairs: these terms carry little calibratable linear signal and commonly show low/unstable R^2.

## Constant term (offset)

- Hamiltonian identity term: `(-6.9817057553201529e+00) * I`
- offset: `-6.9817057553201529e+00`
- E_cdr_unmit: `-7.8705318570523701e+00`
- E_cdr_rem: `-7.8705318570523701e+00`
- delta (unmit - rem): `0.0000000000000000e+00`

## Int encoding map

- `I=0, X=1, Y=2, Z=3`

## Per-term full details (first 20 terms only)

### term 0

- pauli term from int row: `(0.011733624356716977)*X(q(0))*X(q(1))`
- int observable row: `[1, 1, 0, 0, 0, 0]`
- Hamiltonian weight w_0: `1.1733624356716977e-02`
- OGM effective shots used for this term: `427`
- fitted unmit coeffs: `a_u=-2.0608789461944335e-08`, `b_u=-7.9222840473681575e-11`
- fitted rem coeffs: `a_r=-1.8615085277176900e-08`, `b_r=-7.9222840473681252e-11`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=-4.5085198969549501e-18`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=2.6004728132387706e-02`, `x_rem=2.8789874401088766e-02`, `y_exact=1.6775627154198789e-09`
  - train[2] t_remaining=2: `x_unmit=5.0505050505050509e-03`, `x_rem=5.5914218878147662e-03`, `y_exact=-1.2694536807203956e-08`
  - train[3] t_remaining=2: `x_unmit=1.0500000000000000e-01`, `x_rem=1.1624566104766881e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.2406947890818859e-02`, `x_rem=1.3735751039544952e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=3.0612244897959183e-02`, `x_rem=3.3890863279203737e-02`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=4.2128603104212861e-02`, `x_rem=4.6640641112991411e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-3.6036036036036036e-02`, `x_rem=-3.9895550767110689e-02`, `y_exact=7.2148467039304624e-10`
  - train[8] t_remaining=2: `x_unmit=-2.7295285359801490e-02`, `x_rem=-3.0218652286998854e-02`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=2.3496783589950402e-18`, `x_rem=1.1748391794975201e-17`, `y_exact=6.0636777673394572e-09`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=2.0631159987349511e-16`, `SS_tot=1.9946338662182525e-16`, `R^2_manual=nan`, `R^2_model=0.053866`
- target x values: `x_u_target=-5.1597051597051594e-02`, `x_r_target=-5.7123174961999423e-02`
- target contribution to E_cdr_unmit: `1.1547410949054831e-11`
- target contribution to E_cdr_rem: `1.1547410949054844e-11`

### term 1

- pauli term from int row: `(0.0027757464116671326)*X(q(0))*X(q(1))*Z(q(2))*Z(q(3))`
- int observable row: `[1, 1, 3, 3, 0, 0]`
- Hamiltonian weight w_1: `2.7757464116671326e-03`
- OGM effective shots used for this term: `68`
- fitted unmit coeffs: `a_u=1.3665411868257802e-08`, `b_u=5.4502589334529420e-10`
- fitted rem coeffs: `a_r=1.0891891151309627e-08`, `b_r=5.4502589334529389e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.3333333333333334e-02`, `x_rem=1.6728545029102974e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-9.0909090909090912e-02`, `x_rem=-1.1405826156206531e-01`, `y_exact=-1.6775627154198789e-09`
  - train[2] t_remaining=2: `x_unmit=-6.4516129032258063e-02`, `x_rem=-8.0944572721465741e-02`, `y_exact=1.2694536807203956e-08`
  - train[3] t_remaining=2: `x_unmit=4.0000000000000001e-02`, `x_rem=5.0185635087308750e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-5.4054054054054057e-02`, `x_rem=-6.7818425793660431e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=5.7471264367816091e-02`, `x_rem=7.2105797539236668e-02`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-9.9009900990099011e-03`, `x_rem=-1.2422186902799217e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=1.1764705882352941e-02`, `x_rem=1.4760480908031979e-02`, `y_exact=-7.2148467039304624e-10`
  - train[8] t_remaining=2: `x_unmit=-2.4050632911392406e-01`, `x_rem=-3.0174907172748916e-01`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=2.4528301886792453e-01`, `x_rem=3.0774210195047819e-01`, `y_exact=-6.0473936758131918e-09`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=2.8291051487316280e-16`, `SS_tot=1.9925236006714084e-16`, `R^2_manual=nan`, `R^2_model=0.182505`
- target x values: `x_u_target=1.0769230769230770e-01`, `x_r_target=1.3511517138890813e-01`
- target contribution to E_cdr_unmit: `5.5978079092710322e-12`
- target contribution to E_cdr_rem: `5.5978079092710298e-12`

### term 2

- pauli term from int row: `(0.002111137889379712)*X(q(0))*X(q(1))*Z(q(2))*Z(q(3))*X(q(4))*X(q(5))`
- int observable row: `[1, 1, 3, 3, 1, 1]`
- Hamiltonian weight w_2: `2.1111378893797121e-03`
- OGM effective shots used for this term: `42`
- fitted unmit coeffs: `a_u=-2.2852083056684466e-08`, `b_u=-5.1387828227115002e-09`
- fitted rem coeffs: `a_r=-1.5078816176959976e-08`, `b_r=-5.1387828227114994e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.1212121212121213e-01`, `x_rem=3.2147162619339015e-01`, `y_exact=-1.6449162934596245e-08`
  - train[1] t_remaining=2: `x_unmit=-8.5714285714285715e-02`, `x_rem=-1.2990077956385984e-01`, `y_exact=-2.5458010532368623e-08`
  - train[2] t_remaining=2: `x_unmit=3.7037037037037035e-02`, `x_rem=5.6129966478210995e-02`, `y_exact=3.4507496372920286e-11`
  - train[3] t_remaining=2: `x_unmit=4.7619047619047616e-02`, `x_rem=7.2167099757699832e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.8571428571428571e-02`, `x_rem=4.3300259854619834e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=6.6666666666666666e-02`, `x_rem=1.0103393966077998e-01`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-1.8867924528301886e-02`, `x_rem=-2.8594511224749122e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=3.2075471698113206e-01`, `x_rem=4.8610669082073282e-01`, `y_exact=5.5042523507022774e-11`
  - train[8] t_remaining=2: `x_unmit=1.1764705882352941e-01`, `x_rem=1.7829518763667024e-01`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=1.2000000000000000e-01`, `x_rem=1.8186109138940362e-01`, `y_exact=-4.0751227755122654e-09`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=9.4577591329257930e-16`, `SS_tot=7.2468169201211464e-16`, `R^2_manual=nan`, `R^2_model=0.117298`
- target x values: `x_u_target=0.0000000000000000e+00`, `x_r_target=9.6219328800846903e-17`
- target contribution to E_cdr_unmit: `-1.0848679122319876e-11`
- target contribution to E_cdr_rem: `-1.0848679122319878e-11`

### term 3

- pauli term from int row: `(0.002111137889379712)*X(q(0))*X(q(1))*Z(q(2))*Z(q(3))*Y(q(4))*Y(q(5))`
- int observable row: `[1, 1, 3, 3, 2, 2]`
- Hamiltonian weight w_3: `2.1111378893797121e-03`
- OGM effective shots used for this term: `26`
- fitted unmit coeffs: `a_u=3.8154350945447572e-08`, `b_u=3.7176721030294662e-09`
- fitted rem coeffs: `a_r=2.5175930037998680e-08`, `b_r=3.7176721030294682e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-9.5238095238095233e-02`, `x_rem=-1.4433419951539964e-01`, `y_exact=1.6449162934596245e-08`
  - train[1] t_remaining=2: `x_unmit=3.2258064516129031e-02`, `x_rem=4.8887390158441749e-02`, `y_exact=2.5458010532368623e-08`
  - train[2] t_remaining=2: `x_unmit=2.8571428571428571e-02`, `x_rem=4.3300259854619702e-02`, `y_exact=-3.4507496372920286e-11`
  - train[3] t_remaining=2: `x_unmit=-9.0909090909090912e-02`, `x_rem=-1.3777355408288150e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.2820512820512819e-01`, `x_rem=-1.9429603780919180e-01`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=4.7619047619047616e-02`, `x_rem=7.2167099757699721e-02`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-2.0833333333333334e-01`, `x_rem=-3.1573106143993679e-01`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=1.2500000000000000e-01`, `x_rem=1.8943863686396212e-01`, `y_exact=-5.5042523507022774e-11`
  - train[8] t_remaining=2: `x_unmit=1.1111111111111110e-01`, `x_rem=1.6838989943463281e-01`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-7.1428571428571425e-02`, `x_rem=-1.0825064963654976e-01`, `y_exact=9.3268257008494603e-09`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=9.6762841601097711e-16`, `SS_tot=7.4410369175124806e-16`, `R^2_manual=nan`, `R^2_model=0.294300`
- target x values: `x_u_target=-2.0000000000000001e-01`, `x_r_target=-3.0310181898233945e-01`
- target contribution to E_cdr_unmit: `-8.2613007481295398e-12`
- target contribution to E_cdr_rem: `-8.2613007481295431e-12`

### term 4

- pauli term from int row: `(-4.163812615366573e-05)*X(q(0))*X(q(1))*Z(q(3))`
- int observable row: `[1, 1, 0, 3, 0, 0]`
- Hamiltonian weight w_4: `-4.1638126153665730e-05`
- OGM effective shots used for this term: `238`
- fitted unmit coeffs: `a_u=-1.8803159643150354e-08`, `b_u=1.1273409989644041e-09`
- fitted rem coeffs: `a_r=-1.5815624606536984e-08`, `b_r=1.1273409989644041e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.9773755656108594e-02`, `x_rem=5.9175903381908521e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-7.4889867841409691e-02`, `x_rem=-8.9036391271858362e-02`, `y_exact=-1.6775627154198789e-09`
  - train[2] t_remaining=2: `x_unmit=-3.7735849056603772e-02`, `x_rem=-4.4864063970503548e-02`, `y_exact=1.2694536807203956e-08`
  - train[3] t_remaining=2: `x_unmit=2.6315789473684209e-02`, `x_rem=3.1286781453114310e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.7522935779816515e-02`, `x_rem=3.2721954914266355e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=5.3097345132743362e-02`, `x_rem=6.3127311250531545e-02`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-1.5267175572519083e-02`, `x_rem=-1.8151109850661734e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-1.7408906882591094e-01`, `x_rem=-2.0697409268983319e-01`, `y_exact=-7.2148467039304624e-10`
  - train[8] t_remaining=2: `x_unmit=-4.6511627906976744e-03`, `x_rem=-5.5297567219457864e-03`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-6.2801932367149760e-02`, `x_rem=-7.4665072646562650e-02`, `y_exact=-6.0473936758131918e-09`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=2.3599355314519579e-16`, `SS_tot=1.9925236006714084e-16`, `R^2_manual=nan`, `R^2_model=0.107609`
- target x values: `x_u_target=-3.6697247706422013e-02`, `x_r_target=-4.3629273219021804e-02`
- target contribution to E_cdr_unmit: `-7.5671681716896319e-14`
- target contribution to E_cdr_rem: `-7.5671681716896319e-14`

### term 5

- pauli term from int row: `(-0.0015406702306936157)*X(q(0))*X(q(1))*Z(q(3))*Z(q(4))`
- int observable row: `[1, 1, 0, 3, 3, 0]`
- Hamiltonian weight w_5: `-1.5406702306936157e-03`
- OGM effective shots used for this term: `170`
- fitted unmit coeffs: `a_u=-2.0215063674776995e-08`, `b_u=1.9853255042343653e-09`
- fitted rem coeffs: `a_r=-1.5591933153143269e-08`, `b_r=1.9853255042343657e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-8.2191780821917804e-02`, `x_rem=-1.0656228875143878e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=4.3478260869565216e-02`, `x_rem=5.6369906368514744e-02`, `y_exact=1.6775627154198789e-09`
  - train[2] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=0.0000000000000000e+00`, `y_exact=1.2694536807203956e-08`
  - train[3] t_remaining=2: `x_unmit=-3.2679738562091505e-02`, `x_rem=-4.2369537466530674e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=3.0839528461809902e-18`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=6.4748201438848921e-02`, `x_rem=8.3946551210665826e-02`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=4.3478260869565216e-02`, `x_rem=5.6369906368514730e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-1.8518518518518517e-01`, `x_rem=-2.4009404564367384e-01`, `y_exact=-7.2148467039304624e-10`
  - train[8] t_remaining=2: `x_unmit=8.8235294117647065e-02`, `x_rem=1.1439775115963281e-01`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-6.4935064935064929e-02`, `x_rem=-8.4188821199729813e-02`, `y_exact=2.3460571318905191e-09`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=1.8349084883357914e-16`, `SS_tot=1.4440065415257233e-16`, `R^2_manual=nan`, `R^2_model=0.173639`
- target x values: `x_u_target=-4.5751633986928102e-02`, `x_r_target=-5.9317352453142964e-02`
- target contribution to E_cdr_unmit: `-4.4836549595200485e-12`
- target contribution to E_cdr_rem: `-4.4836549595200510e-12`

### term 6

- pauli term from int row: `(0.010540187792814543)*X(q(0))*X(q(1))*Z(q(3))*Z(q(5))`
- int observable row: `[1, 1, 0, 3, 0, 3]`
- Hamiltonian weight w_6: `1.0540187792814543e-02`
- OGM effective shots used for this term: `170`
- fitted unmit coeffs: `a_u=-2.8722232274237136e-08`, `b_u=1.8600565928000409e-09`
- fitted rem coeffs: `a_r=-2.1810481592715265e-08`, `b_r=1.8600565928000417e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.4794520547945202e-02`, `x_rem=7.2158926883081800e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=6.2111801242236021e-03`, `x_rem=8.1795056870573833e-03`, `y_exact=1.6775627154198789e-09`
  - train[2] t_remaining=2: `x_unmit=1.3333333333333334e-02`, `x_rem=1.7558672208216587e-02`, `y_exact=1.2694536807203956e-08`
  - train[3] t_remaining=2: `x_unmit=-2.0261437908496732e-01`, `x_rem=-2.6682296002682043e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=5.5555555555555552e-02`, `x_rem=7.3161134200902367e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=1.5107913669064749e-01`, `x_rem=1.9895617789885686e-01`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-6.2111801242236021e-03`, `x_rem=-8.1795056870573712e-03`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-4.9382716049382713e-02`, `x_rem=-6.5032119289691048e-02`, `y_exact=-7.2148467039304624e-10`
  - train[8] t_remaining=2: `x_unmit=-1.4705882352941177e-01`, `x_rem=-1.9366182582591809e-01`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=1.5584415584415584e-01`, `x_rem=2.0523123360253132e-01`, `y_exact=2.3623404141678379e-09`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=2.7305975665682921e-16`, `SS_tot=1.4442520013927742e-16`, `R^2_manual=nan`, `R^2_model=0.342208`
- target x values: `x_u_target=-3.2679738562091505e-02`, `x_r_target=-4.3035961294648428e-02`
- target contribution to E_cdr_unmit: `2.9498735401195401e-11`
- target contribution to E_cdr_rem: `2.9498735401195407e-11`

### term 7

- pauli term from int row: `(0.00837336167736521)*X(q(0))*Y(q(1))*Z(q(2))*X(q(3))*Z(q(4))*Y(q(5))`
- int observable row: `[1, 2, 3, 1, 3, 2]`
- Hamiltonian weight w_7: `8.3733616773652101e-03`
- OGM effective shots used for this term: `64`
- fitted unmit coeffs: `a_u=1.0588033290096794e+00`, `b_u=2.5537565185739203e-02`
- fitted rem coeffs: `a_r=6.9864531500642191e-01`, `b_r=2.5537565185739158e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.0144927536231885e-01`, `x_rem=1.5374729948379529e-01`, `y_exact=-2.8278177176177497e-09`
  - train[1] t_remaining=2: `x_unmit=4.2424242424242425e-01`, `x_rem=6.4294325238678041e-01`, `y_exact=9.6328932076528639e-01`
  - train[2] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=-2.4004822154057440e-17`, `y_exact=-1.2200242552329321e-10`
  - train[3] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=1.1102230246251566e-17`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.7027027027027029e-02`, `x_rem=4.0959705267883700e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=9.5890410958904104e-02`, `x_rem=1.4532278992303946e-01`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-4.1095890410958902e-02`, `x_rem=-6.2281195681302651e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=9.5890410958904104e-02`, `x_rem=1.4532278992303946e-01`, `y_exact=3.8920937088907981e-11`
  - train[8] t_remaining=2: `x_unmit=1.0769230769230770e-01`, `x_rem=1.6320867175972117e-01`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-1.3513513513513514e-01`, `x_rem=-2.0479852633941842e-01`, `y_exact=1.2344485347353550e-01`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=3.7077487643686885e-01`, `SS_tot=8.2506583143637080e-01`, `R^2_manual=0.550612`, `R^2_model=0.438172`
- target x values: `x_u_target=-1.8421052631578946e-01`, `x_r_target=-2.7917272801004939e-01`
- target contribution to E_cdr_unmit: `-1.4193279548924758e-03`
- target contribution to E_cdr_rem: `-1.4193279548924767e-03`

### term 8

- pauli term from int row: `(-0.00837336167736521)*X(q(0))*Y(q(1))*Z(q(2))*Y(q(3))*Z(q(4))*X(q(5))`
- int observable row: `[1, 2, 3, 2, 3, 1]`
- Hamiltonian weight w_8: `-8.3733616773652101e-03`
- OGM effective shots used for this term: `77`
- fitted unmit coeffs: `a_u=1.7686055349166787e+00`, `b_u=-6.0295921642462204e-02`
- fitted rem coeffs: `a_r=1.1670042369621862e+00`, `b_r=-6.0295921642462190e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=8.7719298245614030e-02`, `x_rem=1.3293939429049964e-01`, `y_exact=-2.8278177176177497e-09`
  - train[1] t_remaining=2: `x_unmit=-3.2307692307692309e-01`, `x_rem=-4.8962601527916366e-01`, `y_exact=-9.5847688540675113e-01`
  - train[2] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=4.9343245538895844e-17`, `y_exact=-1.2200242552329321e-10`
  - train[3] t_remaining=2: `x_unmit=-9.8039215686274508e-02`, `x_rem=-1.4857932303055843e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.3043478260869565e-01`, `x_rem=1.9767509933630828e-01`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=2.2500000000000001e-01`, `x_rem=3.4098954635513190e-01`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=-6.0797927538996674e-17`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-1.3333333333333334e-02`, `x_rem=-2.0206787932155919e-02`, `y_exact=3.8920937088907981e-11`
  - train[8] t_remaining=2: `x_unmit=8.5714285714285715e-02`, `x_rem=1.2990077956385976e-01`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=5.8823529411764705e-02`, `x_rem=8.9147593818335161e-02`, `y_exact=-1.2282814242927334e-01`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=3.6381584655270116e-01`, `SS_tot=8.1684263547979141e-01`, `R^2_manual=0.554607`, `R^2_model=0.548066`
- target x values: `x_u_target=8.8235294117647065e-02`, `x_r_target=1.3372139072750272e-01`
- target contribution to E_cdr_unmit: `-8.0181224704529595e-04`
- target contribution to E_cdr_rem: `-8.0181224704529595e-04`

### term 9

- pauli term from int row: `(-0.003034657024161045)*X(q(0))*Y(q(1))*X(q(3))*Y(q(4))`
- int observable row: `[1, 2, 0, 1, 2, 0]`
- Hamiltonian weight w_9: `-3.0346570241610452e-03`
- OGM effective shots used for this term: `63`
- fitted unmit coeffs: `a_u=1.1908595051888891e-01`, `b_u=-1.8452462156840144e-02`
- fitted rem coeffs: `a_r=9.1851314932329717e-02`, `b_r=-1.8452462156840141e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=3.4482758620689655e-02`, `x_rem=4.4707167119856533e-02`, `y_exact=-6.9344013622967826e-02`
  - train[1] t_remaining=2: `x_unmit=-1.2280701754385964e-01`, `x_rem=-1.5922026184791002e-01`, `y_exact=-2.6801878122779677e-02`
  - train[2] t_remaining=2: `x_unmit=9.8039215686274508e-02`, `x_rem=1.2710861239959204e-01`, `y_exact=8.6116258180269134e-02`
  - train[3] t_remaining=2: `x_unmit=-1.6393442622950821e-02`, `x_rem=-2.1254226991407197e-02`, `y_exact=-9.8943630161376056e-02`
  - train[4] t_remaining=2: `x_unmit=1.4893617021276595e-01`, `x_rem=1.9309691330491216e-01`, `y_exact=-5.8816097149055779e-02`
  - train[5] t_remaining=2: `x_unmit=8.6956521739130432e-02`, `x_rem=1.1273981273702946e-01`, `y_exact=7.5552546747712057e-02`
  - train[6] t_remaining=2: `x_unmit=1.4285714285714285e-01`, `x_rem=1.8521540663940556e-01`, `y_exact=9.6488671865840203e-02`
  - train[7] t_remaining=2: `x_unmit=-3.5714285714285712e-02`, `x_rem=-4.6303851659851389e-02`, `y_exact=-3.9769206120374267e-02`
  - train[8] t_remaining=2: `x_unmit=-7.1428571428571425e-02`, `x_rem=-9.2607703319702778e-02`, `y_exact=-7.3782335815627320e-02`
  - train[9] t_remaining=2: `x_unmit=1.1111111111111110e-01`, `x_rem=1.4405642738620431e-01`, `y_exact=-9.9069863738765085e-02`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=4.9115969235108012e-02`, `SS_tot=5.3708843055398921e-02`, `R^2_manual=0.085514`, `R^2_model=0.018498`
- target x values: `x_u_target=4.7619047619047616e-02`, `x_r_target=6.1738468879801847e-02`
- target contribution to E_cdr_unmit: `3.8788083601080772e-05`
- target contribution to E_cdr_rem: `3.8788083601080772e-05`

### term 10

- pauli term from int row: `(0.003034657024161045)*X(q(0))*Y(q(1))*Y(q(3))*X(q(4))`
- int observable row: `[1, 2, 0, 2, 1, 0]`
- Hamiltonian weight w_10: `3.0346570241610452e-03`
- OGM effective shots used for this term: `53`
- fitted unmit coeffs: `a_u=-2.1287071251397260e-01`, `b_u=-7.6923355301256033e-02`
- fitted rem coeffs: `a_r=-1.6418775489295881e-01`, `b_r=-7.6923355301256047e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.4050632911392406e-01`, `x_rem=-3.1181834282330301e-01`, `y_exact=-9.9833412891856876e-02`
  - train[1] t_remaining=2: `x_unmit=-2.1428571428571427e-01`, `x_rem=-2.7782310995910836e-01`, `y_exact=-9.9833442855232768e-02`
  - train[2] t_remaining=2: `x_unmit=2.0833333333333334e-01`, `x_rem=2.7010580134913309e-01`, `y_exact=-9.9833433213846967e-02`
  - train[3] t_remaining=2: `x_unmit=8.7719298245614030e-02`, `x_rem=1.1372875846279290e-01`, `y_exact=-9.9833424375471630e-02`
  - train[4] t_remaining=2: `x_unmit=8.7719298245614030e-02`, `x_rem=1.1372875846279289e-01`, `y_exact=5.8816097149055779e-02`
  - train[5] t_remaining=2: `x_unmit=7.4626865671641784e-02`, `x_rem=9.6754316901181997e-02`, `y_exact=-7.5552546747712057e-02`
  - train[6] t_remaining=2: `x_unmit=2.0408163265306121e-02`, `x_rem=2.6459343805629369e-02`, `y_exact=-9.6488671865840203e-02`
  - train[7] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=-7.1627291911300420e-18`, `y_exact=-9.9833465492374263e-02`
  - train[8] t_remaining=2: `x_unmit=-8.1967213114754092e-02`, `x_rem=-1.0627113495703598e-01`, `y_exact=7.3782335815627320e-02`
  - train[9] t_remaining=2: `x_unmit=-9.4339622641509441e-02`, `x_rem=-1.2231206098828669e-01`, `y_exact=-9.9833447189278640e-02`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=5.5168412723408507e-02`, `SS_tot=4.2960710152338447e-02`, `R^2_manual=-0.284160`, `R^2_model=0.218856`
- target x values: `x_u_target=-4.7619047619047616e-02`, `x_r_target=-6.1738468879801847e-02`
- target contribution to E_cdr_unmit: `-2.0267459082181654e-04`
- target contribution to E_cdr_rem: `-2.0267459082181654e-04`

### term 11

- pauli term from int row: `(0.0023679367103782043)*X(q(0))*Z(q(1))*X(q(2))*Z(q(3))`
- int observable row: `[1, 3, 1, 3, 0, 0]`
- Hamiltonian weight w_11: `2.3679367103782043e-03`
- OGM effective shots used for this term: `459`
- fitted unmit coeffs: `a_u=-7.0766321855523208e-09`, `b_u=-4.3277660895697464e-10`
- fitted rem coeffs: `a_r=-5.6403647563618478e-09`, `b_r=-4.3277660895697480e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.2605042016806723e-02`, `x_rem=1.5814800972891424e-02`, `y_exact=-1.9389760904059116e-09`
  - train[1] t_remaining=2: `x_unmit=-4.9250535331905779e-02`, `x_rem=-6.1791734850540757e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.3932135728542916e-02`, `x_rem=4.2572644535142118e-02`, `y_exact=-2.5389069762158448e-09`
  - train[3] t_remaining=2: `x_unmit=8.5106382978723406e-03`, `x_rem=1.0677794699427401e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.2123893805309734e-02`, `x_rem=2.7757541530591109e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=5.9642147117296221e-03`, `x_rem=7.4829475776305235e-03`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-6.3829787234042548e-02`, `x_rem=-8.0083460245705437e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-6.5040650406504072e-02`, `x_rem=-8.1602658678550835e-02`, `y_exact=7.2148497447615376e-10`
  - train[8] t_remaining=2: `x_unmit=-1.4492753623188406e-02`, `x_rem=-1.8183201118590123e-02`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-2.5948103792415168e-02`, `x_rem=-3.2555551703343948e-02`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=6.9660426044692177e-18`, `SS_tot=9.3151648187706236e-18`, `R^2_manual=nan`, `R^2_model=0.089605`
- target x values: `x_u_target=-8.0610021786492375e-02`, `x_r_target=-1.0113662844392284e-01`
- target contribution to E_cdr_unmit: `3.2599589682966393e-13`
- target contribution to E_cdr_rem: `3.2599589682966307e-13`

### term 12

- pauli term from int row: `(-0.004788969401730167)*X(q(0))*Z(q(1))*X(q(2))*Z(q(3))*Z(q(4))`
- int observable row: `[1, 3, 1, 3, 3, 0]`
- Hamiltonian weight w_12: `-4.7889694017301671e-03`
- OGM effective shots used for this term: `459`
- fitted unmit coeffs: `a_u=5.9833239939002060e-09`, `b_u=-3.2465370380666128e-11`
- fitted rem coeffs: `a_r=4.3731303532265204e-09`, `b_r=-3.2465370380666102e-11`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-4.2016806722689079e-02`, `x_rem=-5.7487462642280676e-02`, `y_exact=1.9389760904059116e-09`
  - train[1] t_remaining=2: `x_unmit=-1.0064239828693790e-01`, `x_rem=-1.3769909145964696e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=2.1956087824351298e-02`, `x_rem=3.0040354730038101e-02`, `y_exact=-2.5389069762158448e-09`
  - train[3] t_remaining=2: `x_unmit=3.4042553191489362e-02`, `x_rem=4.6577076115277616e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=8.8495575221238937e-03`, `x_rem=1.2107978857400741e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=5.9642147117296221e-03`, `x_rem=8.1602481762600970e-03`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-5.5319148936170209e-02`, `x_rem=-7.5687748687326098e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-1.5447154471544716e-01`, `x_rem=-2.1134821631576683e-01`, `y_exact=7.2148497447615376e-10`
  - train[8] t_remaining=2: `x_unmit=-4.7619047619047616e-02`, `x_rem=-6.5152457661251398e-02`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-9.7804391217564873e-02`, `x_rem=-1.3381612561562412e-01`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=1.5762301214830425e-17`, `SS_tot=1.0724739941790988e-17`, `R^2_manual=nan`, `R^2_model=0.042040`
- target x values: `x_u_target=-2.3965141612200435e-02`, `x_r_target=-3.2789145358930440e-02`
- target contribution to E_cdr_unmit: `8.4217176733330502e-13`
- target contribution to E_cdr_rem: `8.4217176733330522e-13`

### term 13

- pauli term from int row: `(-0.035116770274161394)*X(q(0))*Z(q(1))*X(q(2))*Z(q(3))*Z(q(4))*Z(q(5))`
- int observable row: `[1, 3, 1, 3, 3, 3]`
- Hamiltonian weight w_13: `-3.5116770274161394e-02`
- OGM effective shots used for this term: `459`
- fitted unmit coeffs: `a_u=3.2472754364093967e-09`, `b_u=-3.3403545190726101e-10`
- fitted rem coeffs: `a_r=2.1426961060887554e-09`, `b_r=-3.3403545190726085e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=3.3613445378151259e-02`, `x_rem=5.0941482181905778e-02`, `y_exact=-1.9389760904059116e-09`
  - train[1] t_remaining=2: `x_unmit=2.7837259100642397e-02`, `x_rem=4.2187619344436905e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.9960079840319360e-03`, `x_rem=3.0249682533167372e-03`, `y_exact=-2.5389069762158448e-09`
  - train[3] t_remaining=2: `x_unmit=-4.2553191489361703e-03`, `x_rem=-6.4489748719646226e-03`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-8.8495575221238937e-03`, `x_rem=-1.3411584910722956e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=-2.9821073558648110e-02`, `x_rem=-4.5194108198161963e-02`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=4.2553191489361703e-03`, `x_rem=6.4489748719646417e-03`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=3.9715295189842999e-17`, `y_exact=7.2148497447615376e-10`
  - train[8] t_remaining=2: `x_unmit=-2.6915113871635612e-02`, `x_rem=-4.0790099863047748e-02`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=1.0179640718562874e-01`, `x_rem=1.5427338091915471e-01`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=9.7200973950488159e-18`, `SS_tot=9.3151648187706236e-18`, `R^2_manual=nan`, `R^2_model=0.024784`
- target x values: `x_u_target=5.4466230936819175e-02`, `x_r_target=8.2544068350310279e-02`
- target contribution to E_cdr_unmit: `5.5192535528178961e-12`
- target contribution to E_cdr_rem: `5.5192535528179107e-12`

### term 14

- pauli term from int row: `(-0.03305872840296005)*X(q(0))*Z(q(1))*X(q(2))*Z(q(4))`
- int observable row: `[1, 3, 1, 0, 3, 0]`
- Hamiltonian weight w_14: `-3.3058728402960050e-02`
- OGM effective shots used for this term: `459`
- fitted unmit coeffs: `a_u=1.4456131974042475e-08`, `b_u=-3.1376260375250657e-10`
- fitted rem coeffs: `a_r=1.1346424813418943e-08`, `b_r=-3.1376260375250688e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.0420168067226892e-02`, `x_rem=6.4238790254990130e-02`, `y_exact=-1.9389760904059116e-09`
  - train[1] t_remaining=2: `x_unmit=1.0706638115631691e-02`, `x_rem=1.3640999358215273e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.7964071856287425e-02`, `x_rem=-2.2887473174682083e-02`, `y_exact=2.5389069762158448e-09`
  - train[3] t_remaining=2: `x_unmit=4.2553191489361701e-02`, `x_rem=5.4215716598183153e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=3.9823008849557522e-02`, `x_rem=5.0737274604051925e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=1.3916500994035786e-02`, `x_rem=1.7730587237377977e-02`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-3.4042553191489362e-02`, `x_rem=-4.3372573278546503e-02`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=1.6260162601626018e-02`, `x_rem=2.0716574635078113e-02`, `y_exact=-7.2148497447615376e-10`
  - train[8] t_remaining=2: `x_unmit=4.3478260869565216e-02`, `x_rem=5.5394319132926237e-02`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-9.9800399201596807e-03`, `x_rem=-1.2715262874823344e-02`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=1.6797434262736964e-17`, `SS_tot=1.0724739941790988e-17`, `R^2_manual=nan`, `R^2_model=0.368230`
- target x values: `x_u_target=7.6252723311546838e-02`, `x_r_target=9.7151256867114691e-02`
- target contribution to E_cdr_unmit: `-2.6068686001179942e-11`
- target contribution to E_cdr_rem: `-2.6068686001179981e-11`

### term 15

- pauli term from int row: `(0.007764441381147411)*X(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*X(q(4))*Y(q(5))`
- int observable row: `[1, 3, 2, 3, 1, 2]`
- Hamiltonian weight w_15: `7.7644413811474109e-03`
- OGM effective shots used for this term: `63`
- fitted unmit coeffs: `a_u=-1.3858786342712965e-10`, `b_u=1.8610773745336902e-11`
- fitted rem coeffs: `a_r=-9.1446408267978393e-11`, `b_r=1.8610773745336902e-11`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=3.2258064516129031e-02`, `x_rem=4.8887390158441735e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.8181818181818181e-02`, `x_rem=2.7554710816576251e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-5.8823529411764705e-02`, `x_rem=-8.9147593818335050e-02`, `y_exact=1.7253750804348188e-10`
  - train[3] t_remaining=2: `x_unmit=1.7543859649122806e-02`, `x_rem=2.6587878858099944e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=5.8823529411764705e-02`, `x_rem=8.9147593818335008e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=2.1428571428571427e-01`, `x_rem=3.2475194890964937e-01`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=-2.1739130434782608e-01`, `x_rem=-3.2945849889384715e-01`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=5.3925118338936174e-17`, `y_exact=5.5042500308339244e-11`
  - train[8] t_remaining=2: `x_unmit=-3.1250000000000000e-02`, `x_rem=-4.7359659215990557e-02`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-1.6393442622950821e-02`, `x_rem=-2.4844411391995079e-02`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=2.6868265262080728e-20`, `SS_tot=2.7619602501906608e-20`, `R^2_manual=nan`, `R^2_model=0.040586`
- target x values: `x_u_target=-2.0512820512820512e-01`, `x_r_target=-3.1087366049470699e-01`
- target contribution to E_cdr_unmit: `3.6523197292519104e-13`
- target contribution to E_cdr_rem: `3.6523197292519063e-13`

### term 16

- pauli term from int row: `(-0.007764441381147411)*X(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*Y(q(4))*X(q(5))`
- int observable row: `[1, 3, 2, 3, 2, 1]`
- Hamiltonian weight w_16: `-7.7644413811474109e-03`
- OGM effective shots used for this term: `48`
- fitted unmit coeffs: `a_u=-1.0773616095286448e-10`, `b_u=2.9599568759341917e-11`
- fitted rem coeffs: `a_r=-7.1089089016085232e-11`, `b_r=2.9599568759341911e-11`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.1428571428571428e-01`, `x_rem=1.7320103941847967e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=2.0754716981132076e-01`, `x_rem=3.1453962347223896e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=7.1428571428571425e-02`, `x_rem=1.0825064963654984e-01`, `y_exact=1.7253750804348188e-10`
  - train[3] t_remaining=2: `x_unmit=4.0000000000000001e-02`, `x_rem=6.0620363796467805e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.3333333333333334e-01`, `x_rem=3.5361878881272923e-01`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=-6.6666666666666666e-02`, `x_rem=-1.0103393966077980e-01`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=7.1428571428571425e-02`, `x_rem=1.0825064963654965e-01`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-1.4084507042253521e-02`, `x_rem=-2.1345198519883039e-02`, `y_exact=5.5042500308339244e-11`
  - train[8] t_remaining=2: `x_unmit=4.7619047619047616e-02`, `x_rem=7.2167099757699901e-02`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=-1.4754098360655737e-01`, `x_rem=-2.2359970252795527e-01`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=2.8774831202647310e-20`, `SS_tot=2.7619602501906608e-20`, `R^2_manual=nan`, `R^2_model=0.039318`
- target x values: `x_u_target=-9.3750000000000000e-02`, `x_r_target=-1.4207897764797164e-01`
- target contribution to E_cdr_unmit: `-3.0824703275931303e-13`
- target contribution to E_cdr_rem: `-3.0824703275931293e-13`

### term 17

- pauli term from int row: `(0.00837336167736521)*X(q(0))*X(q(2))*X(q(3))*X(q(4))`
- int observable row: `[1, 0, 1, 1, 1, 0]`
- Hamiltonian weight w_17: `8.3733616773652101e-03`
- OGM effective shots used for this term: `93`
- fitted unmit coeffs: `a_u=1.8174257137868970e+00`, `b_u=-4.1732419701853238e-03`
- fitted rem coeffs: `a_r=1.3988329863379332e+00`, `b_r=-4.1732419701852813e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.3698630136986301e-02`, `x_rem=1.7797866434213754e-02`, `y_exact=-2.3989586368078828e-08`
  - train[1] t_remaining=2: `x_unmit=-6.1728395061728392e-02`, `x_rem=-8.0200262327013064e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.0588235294117647e-01`, `x_rem=-1.3756703820327645e-01`, `y_exact=8.9763918520446850e-09`
  - train[3] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=1.1102230246251566e-17`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=4.5454545454545453e-01`, `x_rem=5.9056556804436866e-01`, `y_exact=8.0802991972244009e-01`
  - train[5] t_remaining=2: `x_unmit=2.9268292682926828e-01`, `x_rem=3.8026660966759357e-01`, `y_exact=6.5366235259282668e-01`
  - train[6] t_remaining=2: `x_unmit=2.5490196078431371e-01`, `x_rem=3.3117990678566561e-01`, `y_exact=2.5667893383486629e-01`
  - train[7] t_remaining=2: `x_unmit=4.2553191489361701e-02`, `x_rem=5.5286989348834553e-02`, `y_exact=5.1016685851250941e-10`
  - train[8] t_remaining=2: `x_unmit=3.4117647058823530e-01`, `x_rem=4.4327156754389080e-01`, `y_exact=6.7364553806606509e-01`
  - train[9] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=2.0185873175002847e-17`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=1.1836343475523536e-01`, `SS_tot=1.0276948048719954e+00`, `R^2_manual=0.884826`, `R^2_model=0.747705`
- target x values: `x_u_target=-1.1494252873563218e-02`, `x_r_target=-1.4933841950547200e-02`
- target contribution to E_cdr_unmit: `-2.0986317729480139e-04`
- target contribution to E_cdr_rem: `-2.0986317729480033e-04`

### term 18

- pauli term from int row: `(0.00837336167736521)*X(q(0))*X(q(2))*Y(q(3))*Y(q(4))`
- int observable row: `[1, 0, 1, 2, 2, 0]`
- Hamiltonian weight w_18: `8.3733616773652101e-03`
- OGM effective shots used for this term: `96`
- fitted unmit coeffs: `a_u=1.4804796188869929e+00`, `b_u=1.0228747338174249e-01`
- fitted rem coeffs: `a_r=1.1394929161561143e+00`, `b_r=1.0228747338174246e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.0000000000000001e-02`, `x_rem=5.1969769987904514e-02`, `y_exact=-2.3989586368078828e-08`
  - train[1] t_remaining=2: `x_unmit=-9.5652173913043481e-02`, `x_rem=-1.2427553692759762e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.1111111111111110e-01`, `x_rem=-1.4436047218862341e-01`, `y_exact=8.9763918520446850e-09`
  - train[3] t_remaining=2: `x_unmit=-3.0434782608695654e-01`, `x_rem=-3.9542216295144689e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=3.9534883720930231e-01`, `x_rem=5.1365470336882280e-01`, `y_exact=8.0802991972244009e-01`
  - train[5] t_remaining=2: `x_unmit=2.1428571428571427e-01`, `x_rem=2.7840948207805949e-01`, `y_exact=6.5366235259282668e-01`
  - train[6] t_remaining=2: `x_unmit=1.0344827586206896e-01`, `x_rem=1.3440457755492533e-01`, `y_exact=2.5667893383486629e-01`
  - train[7] t_remaining=2: `x_unmit=-1.0679611650485436e-01`, `x_rem=-1.3875424025896824e-01`, `y_exact=5.1016685851250941e-10`
  - train[8] t_remaining=2: `x_unmit=2.6213592233009708e-01`, `x_rem=3.4057858972655819e-01`, `y_exact=6.7364553806606509e-01`
  - train[9] t_remaining=2: `x_unmit=-1.0843373493975904e-01`, `x_rem=-1.4088190659371688e-01`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=2.6222915215394421e-01`, `SS_tot=1.0276948048719954e+00`, `R^2_manual=0.744838`, `R^2_model=0.510532`
- target x values: `x_u_target=3.9215686274509803e-02`, `x_r_target=5.0950754890102395e-02`
- target contribution to E_cdr_unmit: `1.3426308451758083e-03`
- target contribution to E_cdr_rem: `1.3426308451758081e-03`

### term 19

- pauli term from int row: `(0.0044790745997579165)*X(q(0))*X(q(2))*Z(q(3))*Z(q(4))`
- int observable row: `[1, 0, 1, 3, 3, 0]`
- Hamiltonian weight w_19: `4.4790745997579165e-03`
- OGM effective shots used for this term: `629`
- fitted unmit coeffs: `a_u=1.6213312215431752e-10`, `b_u=-3.7691524185951775e-10`
- fitted rem coeffs: `a_r=1.2479033268151791e-10`, `b_r=-3.7691524185951770e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-3.2154340836012860e-02`, `x_rem=-4.1776342434006825e-02`, `y_exact=-1.9389760904059116e-09`
  - train[1] t_remaining=2: `x_unmit=-6.3694267515923570e-03`, `x_rem=-8.2754410808765187e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.2258064516129031e-02`, `x_rem=4.1911104828955277e-02`, `y_exact=-2.5389069762158448e-09`
  - train[3] t_remaining=2: `x_unmit=4.0128410914927769e-02`, `x_rem=5.2136607130722712e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=3.3557046979865772e-02`, `x_rem=4.3598800325423251e-02`, `y_exact=0.0000000000000000e+00`
  - train[5] t_remaining=2: `x_unmit=-6.2305295950155761e-03`, `x_rem=-8.0949797488947331e-03`, `y_exact=0.0000000000000000e+00`
  - train[6] t_remaining=2: `x_unmit=1.5847860538827259e-03`, `x_rem=2.0590241675081632e-03`, `y_exact=0.0000000000000000e+00`
  - train[7] t_remaining=2: `x_unmit=-4.8929663608562692e-02`, `x_rem=-6.3571584083063556e-02`, `y_exact=7.2148497447615376e-10`
  - train[8] t_remaining=2: `x_unmit=-2.1001615508885300e-02`, `x_rem=-2.7286228184279327e-02`, `y_exact=0.0000000000000000e+00`
  - train[9] t_remaining=2: `x_unmit=1.3740458015267175e-02`, `x_rem=1.7852211064547287e-02`, `y_exact=0.0000000000000000e+00`
- explicit R^2 calculation for this term (REM branch):
  - formula: `R^2 = 1 - SS_res/SS_tot`, `SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, `SS_tot = sum_i(y_exact_i - mean(y_exact))^2`
  - values: `SS_res=9.3323724570542337e-18`, `SS_tot=9.3151648187706236e-18`, `R^2_manual=nan`, `R^2_model=0.000024`
- target x values: `x_u_target=-6.5359477124183009e-03`, `x_r_target=-8.4917924816836840e-03`
- target contribution to E_cdr_unmit: `-1.6929779328015115e-12`
- target contribution to E_cdr_rem: `-1.6929779328015111e-12`

## Complete expanded energy expressions (first 20 terms only)

### E_cdr_unmit

`E_cdr_unmit = -6.9817057553201529e+00 + 1.1547410949054831e-11 + 5.5978079092710322e-12 + -1.0848679122319876e-11 + -8.2613007481295398e-12 + -7.5671681716896319e-14 + -4.4836549595200485e-12 + 2.9498735401195401e-11 + -1.4193279548924758e-03 + -8.0181224704529595e-04 + 3.8788083601080772e-05 + -2.0267459082181654e-04 + 3.2599589682966393e-13 + 8.4217176733330502e-13 + 5.5192535528178961e-12 + -2.6068686001179942e-11 + 3.6523197292519104e-13 + -3.0824703275931303e-13 + -2.0986317729480139e-04 + 1.3426308451758083e-03 + -1.6929779328015115e-12`

### E_cdr_rem

`E_cdr_rem = -6.9817057553201529e+00 + 1.1547410949054844e-11 + 5.5978079092710298e-12 + -1.0848679122319878e-11 + -8.2613007481295431e-12 + -7.5671681716896319e-14 + -4.4836549595200510e-12 + 2.9498735401195407e-11 + -1.4193279548924767e-03 + -8.0181224704529595e-04 + 3.8788083601080772e-05 + -2.0267459082181654e-04 + 3.2599589682966307e-13 + 8.4217176733330522e-13 + 5.5192535528179107e-12 + -2.6068686001179981e-11 + 3.6523197292519063e-13 + -3.0824703275931293e-13 + -2.0986317729480033e-04 + 1.3426308451758081e-03 + -1.6929779328015111e-12`

## Headline values (target circuit)

- raw finite-shot (unmit / REM): `-7.632159603738 / -7.691288524438 Eh`
- cdr corrected (unmit / REM): `-7.870531857052 / -7.870531857052 Eh`
- reference exact noiseless: `-7.879817546698 Eh`
- Energy error (exact - cdr_rem): `-0.009285689645 Eh`

