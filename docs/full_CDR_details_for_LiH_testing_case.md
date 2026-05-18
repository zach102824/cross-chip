# full CDR details for LiH testing case

## Configuration used

- bond_length: `2.2`
- target params [theta1, theta2, theta3]: `[-0.444980732142, 0.476365247616, 0.1426860331]`
- measurement_scheme: `ogm`
- num_shots: `8192`
- sampling_seed: `1234`
- apply_readout_noise: `True`
- apply_rem in CDR training/target estimation: `True`
- readout p_0_success: `[0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514]`
- readout p_1_success: `[0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514]`
- base_noise_cfg: `{'two_qubit_depol_prob': 0.01, 'one_qubit_depol_prob': 0.001}`
- cdr_cfg: `{"num_circuits": 5, "t_max": 2, "seed": 42, "cdr_fit_scope": "per_pauli"}`
- training resolvers count: `5`
- training non-Clifford counts (t_remaining): `[2, 2, 2, 2, 2]`
- OGM file: `/Users/zacharyhe/shadowgrouping/haozhaowu/LiH/hamil_class/ogm_outputs/OGM_ogm_LiH_2.2.txt`

## Important note on epsilon

- For OGM `SettingSampler`, shot allocation is sampled from file probabilities `p` and total `N_samples` only; epsilon is not used in the sampler math.

## Model definition used

- Per-pauli CDR applies, for each term `k`:
  - unmit branch: `y_k ~= a_u[k] * x_u[k] + b_u[k]`
  - rem branch:   `y_k ~= a_r[k] * x_r[k] + b_r[k]`
- Total corrected energies:
  - `E_cdr_unmit = offset + sum_k w_k * (a_u[k] * x_u_target[k] + b_u[k])`
  - `E_cdr_rem   = offset + sum_k w_k * (a_r[k] * x_r_target[k] + b_r[k])`

## Constant term (offset)

- Hamiltonian identity term: `(-7.1586579073350247e+00) * I`
- offset: `-7.1586579073350247e+00`
- E_cdr_unmit: `-7.8434787927100000`
- E_cdr_rem: `-7.8434787927100000`
- delta (unmit - rem): `0.0000000000000000e+00`

## Int encoding map

- `I=0, X=1, Y=2, Z=3`

## Per-term full details (Pauli term, training pairs, fits, shots, target contribution)

### term 0

- pauli term from int row: `(1.5264219777698207e-02)*X(q(0))*X(q(1))`
- int observable row: `[1, 1, 0, 0, 0, 0]`
- Hamiltonian weight w_0: `1.5264219777698207e-02`
- OGM effective shots used for this term: `574`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-7.0921985815602835e-03`, `x_rem=-7.8517839275696506e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-2.4118738404452691e-02`, `x_rem=-2.6701892280584741e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=6.5486725663716813e-02`, `x_rem=7.2500454425010463e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=3.7567084078711989e-02`, `x_rem=4.1590576403459339e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=5.3604436229205174e-02`, `x_rem=5.9345553567157740e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=7.0921985815602835e-03`, `x_r_target=7.8517839275696662e-03`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 1

- pauli term from int row: `(2.3326675518445836e-03)*X(q(0))*X(q(1))*Z(q(2))*Z(q(3))`
- int observable row: `[1, 1, 3, 3, 0, 0]`
- Hamiltonian weight w_1: `2.3326675518445836e-03`
- OGM effective shots used for this term: `117`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-9.3457943925233638e-03`, `x_rem=-1.1725615674604881e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.9230769230769232e-02`, `x_rem=2.4127709176590760e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=8.6206896551724144e-02`, `x_rem=1.0815869630885500e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.7241379310344827e-02`, `x_rem=2.1631739261770935e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-2.3255813953488372e-02`, `x_rem=-2.9177694818202822e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=6.7796610169491525e-02`, `x_r_target=8.5060398453065608e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 2

- pauli term from int row: `(4.7092490952951533e-03)*X(q(0))*X(q(1))*Z(q(2))*Z(q(3))*X(q(4))*X(q(5))`
- int observable row: `[1, 1, 3, 3, 1, 1]`
- Hamiltonian weight w_2: `4.7092490952951533e-03`
- OGM effective shots used for this term: `49`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-7.4074074074074070e-02`, `x_rem=-1.1225993295642203e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-3.3333333333333333e-02`, `x_rem=-5.0516969830389960e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.2280701754385964e-01`, `x_rem=1.8611515200669940e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.1538461538461539e-01`, `x_rem=1.7486643402827268e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=4.7619047619047616e-02`, `x_rem=7.2167099757699832e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000001e-01`, `x_r_target=1.5155090949116964e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 3

- pauli term from int row: `(4.7092490952951533e-03)*X(q(0))*X(q(1))*Z(q(2))*Z(q(3))*Y(q(4))*Y(q(5))`
- int observable row: `[1, 1, 3, 3, 2, 2]`
- Hamiltonian weight w_3: `4.7092490952951533e-03`
- OGM effective shots used for this term: `68`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=9.4339622641509441e-02`, `x_rem=1.4297255612374499e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=1.0597583416876495e-16`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.8644067796610170e-01`, `x_rem=-2.8255254311913003e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=3.1250000000000000e-02`, `x_rem=4.7359659215990460e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-4.5454545454545456e-02`, `x_rem=-6.8886777041440750e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=6.8965517241379309e-02`, `x_r_target=1.0451786861459987e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 4

- pauli term from int row: `(4.2983656068340519e-03)*X(q(0))*X(q(1))*Z(q(3))`
- int observable row: `[1, 1, 0, 3, 0, 0]`
- Hamiltonian weight w_4: `4.2983656068340519e-03`
- OGM effective shots used for this term: `309`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.3422818791946308e-02`, `x_rem=-1.5958358325078446e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-5.0505050505050504e-02`, `x_rem=-6.0045338142340608e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=2.2653721682847898e-02`, `x_rem=2.6932957496855692e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.5673981191222569e-02`, `x_rem=-1.8634760113140185e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-4.4444444444444446e-02`, `x_rem=-5.2839897565259746e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=6.6666666666666671e-03`, `x_r_target=7.9259846347889511e-03`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 5

- pauli term from int row: `(-3.8063028881711678e-03)*X(q(0))*X(q(1))*Z(q(3))*Z(q(4))`
- int observable row: `[1, 1, 0, 3, 3, 0]`
- Hamiltonian weight w_5: `-3.8063028881711678e-03`
- OGM effective shots used for this term: `192`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-3.6649214659685861e-02`, `x_rem=-4.7515994373460053e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=2.5906735751295335e-02`, `x_rem=3.3588286178130533e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=4.6632124352331605e-02`, `x_rem=6.0458915120634983e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.3300492610837439e-01`, `x_rem=1.7244193031944655e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=4.3478260869565216e-02`, `x_rem=5.6369906368514730e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0989010989010990e-02`, `x_r_target=1.4247338972261965e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 6

- pauli term from int row: `(1.2439498134905486e-02)*X(q(0))*X(q(1))*Z(q(3))*Z(q(5))`
- int observable row: `[1, 1, 0, 3, 0, 3]`
- Hamiltonian weight w_6: `1.2439498134905486e-02`
- OGM effective shots used for this term: `192`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-3.6649214659685861e-02`, `x_rem=-4.8263366017349194e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-4.6632124352331605e-02`, `x_rem=-6.1409863940653814e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.1917098445595854e-01`, `x_rem=-1.5693631895944857e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.4778325123152709e-02`, `x_rem=-1.9461582496791707e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.0869565217391304e-02`, `x_rem=-1.4314134952350449e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=4.3956043956043959e-02`, `x_r_target=5.7885732554560136e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 7

- pauli term from int row: `(1.1507931652465444e-02)*X(q(0))*Y(q(1))*Z(q(2))*X(q(3))*Z(q(4))*Y(q(5))`
- int observable row: `[1, 2, 3, 1, 3, 2]`
- Hamiltonian weight w_7: `1.1507931652465444e-02`
- OGM effective shots used for this term: `85`
- fitted unmit coeffs: `a_u=2.4030881862280000e+00`, `b_u=1.4196427550099999e-01`
- fitted rem coeffs: `a_r=1.5856639820220000e+00`, `b_r=1.4196427550099999e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.0000000000000001e-01`, `x_rem=-3.0310181898233934e-01`, `y_exact=-4.3044022968427953e-01`
  - train[1] t_remaining=2: `x_unmit=-1.6831683168316833e-01`, `x_rem=-2.5508568924256275e-01`, `y_exact=-4.3044022968427953e-01`
  - train[2] t_remaining=2: `x_unmit=-5.5555555555555552e-02`, `x_rem=-8.4194949717316572e-02`, `y_exact=-4.3044022968427953e-01`
  - train[3] t_remaining=2: `x_unmit=-2.2105263157894736e-01`, `x_rem=-3.3500727361205918e-01`, `y_exact=-4.3044022968427953e-01`
  - train[4] t_remaining=2: `x_unmit=-1.7894736842105263e-01`, `x_rem=-2.7119636435261946e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-2.4528301886792453e-01`, `x_r_target=-3.7172864592173704e-01`
- target contribution to E_cdr_unmit: `-3.6212695980335891e-03`
- target contribution to E_cdr_rem: `-3.6212695980335865e-03`

### term 8

- pauli term from int row: `(-1.1507931652465444e-02)*X(q(0))*Y(q(1))*Z(q(2))*Y(q(3))*Z(q(4))*X(q(5))`
- int observable row: `[1, 2, 3, 2, 3, 1]`
- Hamiltonian weight w_8: `-1.1507931652465444e-02`
- OGM effective shots used for this term: `115`
- fitted unmit coeffs: `a_u=1.8962285010060000e+00`, `b_u=5.9360723009999998e-02`
- fitted rem coeffs: `a_r=1.2512155204950000e+00`, `b_r=5.9360723009999998e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.7757009345794392e-01`, `x_rem=2.6910909161983393e-01`, `y_exact=4.3044022968427953e-01`
  - train[1] t_remaining=2: `x_unmit=3.7190082644628097e-01`, `x_rem=5.6361908488451529e-01`, `y_exact=4.3044022968427953e-01`
  - train[2] t_remaining=2: `x_unmit=1.0416666666666667e-01`, `x_rem=1.5786553071996837e-01`, `y_exact=4.3044022968427953e-01`
  - train[3] t_remaining=2: `x_unmit=2.0754716981132076e-01`, `x_rem=3.1453962347223902e-01`, `y_exact=4.3044022968427953e-01`
  - train[4] t_remaining=2: `x_unmit=-2.8571428571428571e-02`, `x_rem=-4.3300259854619910e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.7894736842105263e-01`, `x_r_target=2.7119636435261935e-01`
- target contribution to E_cdr_unmit: `-4.1025047692791559e-03`
- target contribution to E_cdr_rem: `-4.1025047692791567e-03`

### term 9

- pauli term from int row: `(-5.7263977866494889e-03)*X(q(0))*Y(q(1))*X(q(3))*Y(q(4))`
- int observable row: `[1, 2, 0, 1, 2, 0]`
- Hamiltonian weight w_9: `-5.7263977866494889e-03`
- OGM effective shots used for this term: `95`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.4945054945054944e-02`, `x_rem=-7.1236694861309846e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=4.9504950495049507e-02`, `x_rem=6.4183556756229651e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.8681318681318682e-01`, `x_rem=-2.4220476252845338e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.0909090909090909e-01`, `x_rem=-1.4143721961554606e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.5384615384615385e-01`, `x_rem=1.9946274561166749e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-2.3255813953488372e-02`, `x_r_target=-3.0151345266879979e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 10

- pauli term from int row: `(5.7263977866494889e-03)*X(q(0))*Y(q(1))*Y(q(3))*X(q(4))`
- int observable row: `[1, 2, 0, 2, 1, 0]`
- Hamiltonian weight w_10: `5.7263977866494889e-03`
- OGM effective shots used for this term: `106`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.3043478260869565e-01`, `x_rem=-1.6910971910554420e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.2987012987012988e-02`, `x_rem=1.6837764239945940e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-9.3023255813953487e-02`, `x_rem=-1.2060538106751989e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.1827956989247312e-01`, `x_rem=-1.5335039044337881e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.4666666666666667e-01`, `x_rem=1.9015448414978969e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-5.2631578947368418e-02`, `x_r_target=-6.8237255077675751e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 11

- pauli term from int row: `(4.5870037211547705e-03)*X(q(0))*Z(q(1))*X(q(2))*Z(q(3))`
- int observable row: `[1, 3, 1, 3, 0, 0]`
- Hamiltonian weight w_11: `4.5870037211547705e-03`
- OGM effective shots used for this term: `414`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.7142857142857141e-02`, `x_rem=-7.1693764410441077e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=9.6153846153846159e-03`, `x_rem=1.2063854588295371e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.8167938931297711e-02`, `x_rem=4.7887056381019741e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.2285012285012284e-02`, `x_rem=1.5413278589468308e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.3054830287206266e-02`, `x_rem=-1.6379123723011989e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-7.6167076167076173e-02`, `x_r_target=-9.5562327254703414e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 12

- pauli term from int row: `(-1.6982826390582351e-02)*X(q(0))*Z(q(1))*X(q(2))*Z(q(3))*Z(q(4))`
- int observable row: `[1, 3, 1, 3, 3, 0]`
- Hamiltonian weight w_12: `-1.6982826390582351e-02`
- OGM effective shots used for this term: `414`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.7142857142857141e-02`, `x_rem=-7.8182949193501675e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.9230769230769232e-02`, `x_rem=-2.6311569440120738e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=7.6335877862595417e-03`, `x_rem=1.0444287106002119e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-2.7027027027027029e-02`, `x_rem=-3.6978421915845390e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-4.4386422976501305e-02`, `x_rem=-6.0729575417928852e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-4.6683046683046681e-02`, `x_r_target=-6.3871819672823862e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 13

- pauli term from int row: `(-2.3610110167205777e-02)*X(q(0))*Z(q(1))*X(q(2))*Z(q(3))*Z(q(4))*Z(q(5))`
- int observable row: `[1, 3, 1, 3, 3, 3]`
- Hamiltonian weight w_13: `-2.3610110167205777e-02`
- OGM effective shots used for this term: `414`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.7619047619047623e-03`, `x_rem=7.2167099757699344e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.9230769230769232e-02`, `x_rem=-2.9144405671378763e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.8167938931297711e-02`, `x_rem=-5.7843858584415951e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-3.6855036855036855e-02`, `x_rem=-5.5854143547114148e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-9.1383812010443863e-02`, `x_rem=-1.3849299822952843e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-7.1253071253071260e-02`, `x_r_target=-1.0798467752442066e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 14

- pauli term from int row: `(-2.6709685222962787e-02)*X(q(0))*Z(q(1))*X(q(2))*Z(q(4))`
- int observable row: `[1, 3, 1, 0, 3, 0]`
- Hamiltonian weight w_14: `-2.6709685222962787e-02`
- OGM effective shots used for this term: `414`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.7619047619047623e-03`, `x_rem=6.0669968574157650e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=2.4038461538461540e-02`, `x_rem=3.0626666828300585e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.7811704834605598e-02`, `x_rem=2.2693347023921503e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-2.7027027027027029e-02`, `x_rem=-3.4434306488035200e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=6.5274151436031339e-02`, `x_rem=8.3163795042904970e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.7027027027027029e-02`, `x_r_target=3.4434306488035263e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 15

- pauli term from int row: `(1.0106830583060904e-02)*X(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*X(q(4))*Y(q(5))`
- int observable row: `[1, 3, 2, 3, 1, 2]`
- Hamiltonian weight w_15: `1.0106830583060904e-02`
- OGM effective shots used for this term: `97`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.7073170731707318e-01`, `x_rem=-2.5874545522882630e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.1428571428571428e-01`, `x_rem=-1.7320103941847961e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.1951219512195122e-01`, `x_rem=-3.3267272815134802e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-3.0927835051546393e-02`, `x_rem=-4.6871415306547357e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-4.8780487804878050e-02`, `x_rem=-7.3927272922521839e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.2987012987012988e-02`, `x_r_target=1.9681936297554603e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 16

- pauli term from int row: `(-1.0106830583060904e-02)*X(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*Y(q(4))*X(q(5))`
- int observable row: `[1, 3, 2, 3, 2, 1]`
- Hamiltonian weight w_16: `-1.0106830583060904e-02`
- OGM effective shots used for this term: `97`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.7808219178082191e-01`, `x_rem=2.6988518128564470e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=4.0816326530612242e-02`, `x_rem=6.1857514078028399e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.1578947368421054e-02`, `x_rem=-4.7858181944579921e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.1392405063291139e-01`, `x_rem=1.7265293486335787e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.0810810810810811e-01`, `x_rem=1.6383882107153483e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000001e-01`, `x_r_target=1.5155090949116964e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 17

- pauli term from int row: `(1.1507931652465444e-02)*X(q(0))*X(q(2))*X(q(3))*X(q(4))`
- int observable row: `[1, 0, 1, 1, 1, 0]`
- Hamiltonian weight w_17: `1.1507931652465444e-02`
- OGM effective shots used for this term: `126`
- fitted unmit coeffs: `a_u=1.4841704519140000e+00`, `b_u=2.7012707468999999e-02`
- fitted rem coeffs: `a_r=1.1423336699460001e+00`, `b_r=2.7012707468999999e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.2735042735042736e-02`, `x_rem=5.5523258534085997e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.4285714285714285e-01`, `x_rem=1.8560632138537300e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-6.8965517241379309e-02`, `x_rem=-8.9603051703283546e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=6.6666666666666666e-02`, `x_rem=8.6616283313174086e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.4285714285714285e-02`, `x_rem=-1.8560632138537286e-02`, `y_exact=-1.2637056219494841e-01`
- target x values: `x_u_target=-2.5641025641025640e-02`, `x_r_target=-3.3313955120451620e-02`
- target contribution to E_cdr_unmit: `-4.5118710136187763e-04`
- target contribution to E_cdr_rem: `-4.5118710136187774e-04`

### term 18

- pauli term from int row: `(1.1507931652465444e-02)*X(q(0))*X(q(2))*Y(q(3))*Y(q(4))`
- int observable row: `[1, 0, 1, 2, 2, 0]`
- Hamiltonian weight w_18: `1.1507931652465444e-02`
- OGM effective shots used for this term: `139`
- fitted unmit coeffs: `a_u=1.8674615277960001e+00`, `b_u=8.5334342679000003e-02`
- fitted rem coeffs: `a_r=1.4373444625450000e+00`, `b_r=8.5334342679000003e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=6.0402684563758392e-02`, `x_rem=7.8477840585761766e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=8.6206896551724144e-02`, `x_rem=1.1200381462910440e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.4285714285714285e-02`, `x_rem=-1.8560632138537286e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.6666666666666666e-02`, `x_rem=2.1654070828293469e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.2977099236641221e-01`, `x_rem=-1.6860421560961367e-01`, `y_exact=-1.4220230542299955e-01`
- target x values: `x_u_target=-1.4285714285714285e-01`, `x_r_target=-1.8560632138537295e-01`
- target contribution to E_cdr_unmit: `-1.4596047865139635e-03`
- target contribution to E_cdr_rem: `-1.4596047865139629e-03`

### term 19

- pauli term from int row: `(9.2962528164499229e-03)*X(q(0))*X(q(2))*Z(q(3))*Z(q(4))`
- int observable row: `[1, 0, 1, 3, 3, 0]`
- Hamiltonian weight w_19: `9.2962528164499229e-03`
- OGM effective shots used for this term: `606`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-9.0016366612111293e-02`, `x_rem=-1.1695324669945757e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-5.7471264367816091e-02`, `x_rem=-7.4669209752736221e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.0716723549488054e-02`, `x_rem=-3.9908526441223495e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-3.2786885245901641e-02`, `x_rem=-4.2598172121233083e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-2.6455026455026454e-02`, `x_rem=-3.4371540997291242e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-4.5840407470288627e-02`, `x_r_target=-5.9557885809567848e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 20

- pauli term from int row: `(-3.2798138198784431e-02)*X(q(0))*Y(q(2))*X(q(3))*Y(q(5))`
- int observable row: `[1, 0, 2, 1, 0, 2]`
- Hamiltonian weight w_20: `-3.2798138198784431e-02`
- OGM effective shots used for this term: `277`
- fitted unmit coeffs: `a_u=2.2741797590620001e+00`, `b_u=3.6280269490000003e-02`
- fitted rem coeffs: `a_r=1.7232814027150001e+00`, `b_r=3.6280269490000003e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.8985507246376813e-01`, `x_rem=3.8251590124530743e-01`, `y_exact=4.1389769175218394e-01`
  - train[1] t_remaining=2: `x_unmit=4.1095890410958902e-02`, `x_rem=5.4233418875190828e-02`, `y_exact=4.1389769175218394e-01`
  - train[2] t_remaining=2: `x_unmit=2.5925925925925924e-01`, `x_rem=3.4213922278052494e-01`, `y_exact=4.1389769175218394e-01`
  - train[3] t_remaining=2: `x_unmit=2.0430107526881722e-01`, `x_rem=2.6961201426483761e-01`, `y_exact=4.1389769175218394e-01`
  - train[4] t_remaining=2: `x_unmit=2.1481481481481482e-01`, `x_rem=2.8348678458957777e-01`, `y_exact=4.5855203824909219e-01`
- target x values: `x_u_target=2.2764227642276422e-01`, `x_r_target=3.0041492731948527e-01`
- target contribution to E_cdr_unmit: `-1.3881182339500770e-02`
- target contribution to E_cdr_rem: `-1.3881182339500770e-02`

### term 21

- pauli term from int row: `(3.2798138198784431e-02)*X(q(0))*Y(q(2))*Y(q(3))*X(q(5))`
- int observable row: `[1, 0, 2, 2, 0, 1]`
- Hamiltonian weight w_21: `3.2798138198784431e-02`
- OGM effective shots used for this term: `265`
- fitted unmit coeffs: `a_u=2.3170059349090000e+00`, `b_u=4.1163791799999998e-04`
- fitted rem coeffs: `a_r=1.7557333459230000e+00`, `b_r=4.1163791799999998e-04`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.3437500000000000e-01`, `x_rem=-3.0929996702257279e-01`, `y_exact=-4.1389769175218394e-01`
  - train[1] t_remaining=2: `x_unmit=-2.2264150943396227e-01`, `x_rem=-2.9381551584332949e-01`, `y_exact=-4.1389769175218394e-01`
  - train[2] t_remaining=2: `x_unmit=-1.8750000000000000e-01`, `x_rem=-2.4743997361805825e-01`, `y_exact=-4.1389769175218394e-01`
  - train[3] t_remaining=2: `x_unmit=-1.6788321167883211e-01`, `x_rem=-2.2155209316653390e-01`, `y_exact=-4.1389769175218394e-01`
  - train[4] t_remaining=2: `x_unmit=-2.1052631578947367e-01`, `x_rem=-2.7782733879922333e-01`, `y_exact=-4.5389205225939810e-01`
- target x values: `x_u_target=-1.5824915824915825e-01`, `x_r_target=-2.0883822689200873e-01`
- target contribution to E_cdr_unmit: `-1.3712372165613327e-02`
- target contribution to E_cdr_rem: `-1.3712372165613324e-02`

### term 22

- pauli term from int row: `(-1.1507931652465444e-02)*Y(q(0))*X(q(1))*Z(q(2))*X(q(3))*Z(q(4))*Y(q(5))`
- int observable row: `[2, 1, 3, 1, 3, 2]`
- Hamiltonian weight w_22: `-1.1507931652465444e-02`
- OGM effective shots used for this term: `105`
- fitted unmit coeffs: `a_u=9.4805694775299998e-01`, `b_u=4.8549541898000002e-02`
- fitted rem coeffs: `a_r=6.2556994935700005e-01`, `b_r=4.8549541898000002e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.0792079207920791e-01`, `x_rem=3.1510585141728348e-01`, `y_exact=3.8251825715179066e-01`
  - train[1] t_remaining=2: `x_unmit=1.5887850467289719e-01`, `x_rem=2.4078181881774630e-01`, `y_exact=3.8251825715179066e-01`
  - train[2] t_remaining=2: `x_unmit=3.6470588235294116e-01`, `x_rem=5.5271508167367767e-01`, `y_exact=3.8251825715179066e-01`
  - train[3] t_remaining=2: `x_unmit=2.1818181818181817e-01`, `x_rem=3.3065652979891574e-01`, `y_exact=3.8251825715179066e-01`
  - train[4] t_remaining=2: `x_unmit=-7.1428571428571425e-02`, `x_rem=-1.0825064963654976e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=0.0000000000000000e+00`, `x_r_target=-1.8503717077085943e-16`
- target contribution to E_cdr_unmit: `-1.6096278174941763e-03`
- target contribution to E_cdr_rem: `-1.6096278174941745e-03`

### term 23

- pauli term from int row: `(1.1507931652465444e-02)*Y(q(0))*X(q(1))*Z(q(2))*Y(q(3))*Z(q(4))*X(q(5))`
- int observable row: `[2, 1, 3, 2, 3, 1]`
- Hamiltonian weight w_23: `1.1507931652465444e-02`
- OGM effective shots used for this term: `114`
- fitted unmit coeffs: `a_u=1.2862952309850000e+00`, `b_u=-1.1488450846999999e-01`
- fitted rem coeffs: `a_r=8.4875454413600004e-01`, `b_r=-1.1488450846999999e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-7.5268817204301078e-02`, `x_rem=-1.1407057703636435e-01`, `y_exact=-3.8251825715179066e-01`
  - train[1] t_remaining=2: `x_unmit=-2.2580645161290322e-01`, `x_rem=-3.4221173110909286e-01`, `y_exact=-3.8251825715179066e-01`
  - train[2] t_remaining=2: `x_unmit=-1.6483516483516483e-01`, `x_rem=-2.4980919146896097e-01`, `y_exact=-3.8251825715179066e-01`
  - train[3] t_remaining=2: `x_unmit=-1.7829457364341086e-01`, `x_rem=-2.7020704792999245e-01`, `y_exact=-3.8251825715179066e-01`
  - train[4] t_remaining=2: `x_unmit=-9.1743119266055051e-03`, `x_rem=-1.3903753164327511e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=9.9009900990099011e-03`, `x_r_target=1.5005040543680166e-02`
- target contribution to E_cdr_unmit: `-1.0415374662528039e-03`
- target contribution to E_cdr_rem: `-1.0415374662528039e-03`

### term 24

- pauli term from int row: `(5.7263977866494889e-03)*Y(q(0))*X(q(1))*X(q(3))*Y(q(4))`
- int observable row: `[2, 1, 0, 1, 2, 0]`
- Hamiltonian weight w_24: `5.7263977866494889e-03`
- OGM effective shots used for this term: `96`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.0101010101010102e-02`, `x_rem=1.3096038853291293e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=8.6956521739130432e-02`, `x_rem=1.1273981273702946e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-7.8651685393258425e-02`, `x_rem=-1.0197252725090869e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=9.8591549295774641e-02`, `x_rem=1.2782471725818129e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.2987012987012988e-02`, `x_rem=-1.6837764239945961e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-4.0000000000000001e-02`, `x_r_target=-5.1860313859033551e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 25

- pauli term from int row: `(-5.7263977866494889e-03)*Y(q(0))*X(q(1))*Y(q(3))*X(q(4))`
- int observable row: `[2, 1, 0, 2, 1, 0]`
- Hamiltonian weight w_25: `-5.7263977866494889e-03`
- OGM effective shots used for this term: `99`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.5454545454545456e-02`, `x_rem=5.8932174839810843e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=5.3763440860215055e-02`, `x_rem=6.9704722928808541e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.5384615384615385e-01`, `x_rem=1.9946274561166755e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=2.0481927710843373e-01`, `x_rem=2.6554979988059352e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.2643678160919541e-01`, `x_rem=-1.6392627943947388e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.4285714285714285e-01`, `x_r_target=-1.8521540663940550e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 26

- pauli term from int row: `(1.5264219777698207e-02)*Y(q(0))*Y(q(1))`
- int observable row: `[2, 2, 0, 0, 0, 0]`
- Hamiltonian weight w_26: `1.5264219777698207e-02`
- OGM effective shots used for this term: `575`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.2146507666098807e-02`, `x_rem=-2.4518432605170663e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.6274864376130200e-02`, `x_rem=1.8017927312994404e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=2.3172905525846704e-02`, `x_rem=2.5654759249973600e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.7482517482517484e-02`, `x_rem=1.9354921919358808e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.0526315789473684e-02`, `x_rem=1.1653700355656018e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-3.7735849056603774e-03`, `x_r_target=-4.1777416369332600e-03`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 27

- pauli term from int row: `(2.3326675518445836e-03)*Y(q(0))*Y(q(1))*Z(q(2))*Z(q(3))`
- int observable row: `[2, 2, 3, 3, 0, 0]`
- Hamiltonian weight w_27: `2.3326675518445836e-03`
- OGM effective shots used for this term: `112`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.4793388429752067e-02`, `x_rem=-3.1106798607836075e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.0169491525423729e-01`, `x_rem=-1.2759059767959854e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-7.5630252100840331e-02`, `x_rem=-9.4888805837348461e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-4.2735042735042736e-02`, `x_rem=-5.3617131503535059e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-4.5045045045045043e-02`, `x_rem=-5.6515354828050444e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-5.6179775280898875e-02`, `x_r_target=-7.0485442538355050e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 28

- pauli term from int row: `(4.7092490952951533e-03)*Y(q(0))*Y(q(1))*Z(q(2))*Z(q(3))*X(q(4))*X(q(5))`
- int observable row: `[2, 2, 3, 3, 1, 1]`
- Hamiltonian weight w_28: `4.7092490952951533e-03`
- OGM effective shots used for this term: `69`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.7142857142857141e-02`, `x_rem=8.6600519709240042e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.0714285714285714e-01`, `x_rem=1.6237597445482460e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.2580645161290322e-01`, `x_rem=-3.4221173110909286e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=3.8461538461538464e-02`, `x_rem=5.8288811342757645e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=6.1224489795918366e-02`, `x_rem=9.2786271117042726e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-7.6923076923076927e-02`, `x_r_target=-1.1657762268551505e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 29

- pauli term from int row: `(4.7092490952951533e-03)*Y(q(0))*Y(q(1))*Z(q(2))*Z(q(3))*Y(q(4))*Y(q(5))`
- int observable row: `[2, 2, 3, 3, 2, 2]`
- Hamiltonian weight w_29: `4.7092490952951533e-03`
- OGM effective shots used for this term: `43`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.7647058823529413e-01`, `x_rem=-2.6744278145500527e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=9.6774193548387094e-02`, `x_rem=1.4666217047532554e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.2280701754385964e-01`, `x_rem=1.8611515200669951e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=2.3076923076923078e-01`, `x_rem=3.4973286805654547e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-6.4516129032258063e-02`, `x_rem=-9.7774780316883678e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-2.8000000000000003e-01`, `x_r_target=-4.2434254657527526e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 30

- pauli term from int row: `(4.2983656068340519e-03)*Y(q(0))*Y(q(1))*Z(q(3))`
- int observable row: `[2, 2, 0, 3, 0, 0]`
- Hamiltonian weight w_30: `4.2983656068340519e-03`
- OGM effective shots used for this term: `311`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.8391167192429023e-02`, `x_rem=-3.3754193239637520e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-4.1666666666666664e-02`, `x_rem=-4.9537403967431011e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=7.4918566775244305e-02`, `x_rem=8.9070511368149557e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-7.6923076923076927e-02`, `x_rem=-9.1453668862949547e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.0033444816053512e-02`, `x_rem=1.1928739416906456e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-5.1903114186851208e-02`, `x_r_target=-6.1707492831401929e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 31

- pauli term from int row: `(-3.8063028881711678e-03)*Y(q(0))*Y(q(1))*Z(q(3))*Z(q(4))`
- int observable row: `[2, 2, 0, 3, 3, 0]`
- Hamiltonian weight w_31: `-3.8063028881711678e-03`
- OGM effective shots used for this term: `199`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=9.0630450989808690e-18`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=-1.0449157878825003e-17`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=5.3191489361702128e-02`, `x_rem=6.8963183323182922e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.9230769230769232e-02`, `x_rem=-2.4932843201458440e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.0638297872340425e-02`, `x_rem=-1.3792636664636568e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=0.0000000000000000e+00`, `x_r_target=3.1086244689504386e-17`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 32

- pauli term from int row: `(1.2439498134905486e-02)*Y(q(0))*Y(q(1))*Z(q(3))*Z(q(5))`
- int observable row: `[2, 2, 0, 3, 0, 3]`
- Hamiltonian weight w_32: `1.2439498134905486e-02`
- OGM effective shots used for this term: `199`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.1020408163265307e-02`, `x_rem=6.7188796715114477e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=4.7058823529411764e-02`, `x_rem=6.1971784264293797e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.0638297872340425e-02`, `x_rem=1.4009578889534567e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=5.7692307692307696e-02`, `x_rem=7.5975023977860129e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.1276595744680851e-02`, `x_rem=2.8019157779069059e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.0000000000000000e-02`, `x_r_target=2.6338008312324836e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 33

- pauli term from int row: `(-1.0106830583060904e-02)*Y(q(0))*Z(q(1))*X(q(2))*Z(q(3))*X(q(4))*Y(q(5))`
- int observable row: `[2, 3, 1, 3, 1, 2]`
- Hamiltonian weight w_33: `-1.0106830583060904e-02`
- OGM effective shots used for this term: `77`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.2244897959183673e-01`, `x_rem=1.8557254223408537e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-9.5238095238095233e-02`, `x_rem=-1.4433419951539964e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=2.3809523809523808e-02`, `x_rem=3.6083549878849923e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=3.5294117647058823e-02`, `x_rem=5.3488556291001077e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=8.8607594936708861e-02`, `x_rem=1.3428561600483385e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.1627906976744186e-01`, `x_r_target=1.7622198778042994e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 34

- pauli term from int row: `(1.0106830583060904e-02)*Y(q(0))*Z(q(1))*X(q(2))*Z(q(3))*Y(q(4))*X(q(5))`
- int observable row: `[2, 3, 1, 3, 2, 1]`
- Hamiltonian weight w_34: `1.0106830583060904e-02`
- OGM effective shots used for this term: `76`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-6.0606060606060608e-02`, `x_rem=-9.1849036055254366e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=8.5714285714285715e-02`, `x_rem=1.2990077956385968e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=8.7076315656875022e-18`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.0112359550561797e-01`, `x_rem=-1.5325372869893558e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.4705882352941178e-01`, `x_rem=3.7441989403700743e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.9565217391304349e-01`, `x_r_target=-2.9651264900446239e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 35

- pauli term from int row: `(4.5870037211547705e-03)*Y(q(0))*Z(q(1))*Y(q(2))*Z(q(3))`
- int observable row: `[2, 3, 2, 3, 0, 0]`
- Hamiltonian weight w_35: `4.5870037211547705e-03`
- OGM effective shots used for this term: `392`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.4449877750611247e-03`, `x_rem=3.0675816068037043e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-3.9408866995073892e-02`, `x_rem=-4.9443975455476616e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.5075376884422110e-02`, `x_rem=-1.8914184078131443e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=2.0881670533642691e-02`, `x_rem=2.6198997435369999e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=7.4418604651162790e-02`, `x_rem=9.3368623418248808e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=9.4339622641509441e-02`, `x_r_target=1.1836234690403000e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 36

- pauli term from int row: `(-1.6982826390582351e-02)*Y(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*Z(q(4))`
- int observable row: `[2, 3, 2, 3, 3, 0]`
- Hamiltonian weight w_36: `-1.6982826390582351e-02`
- OGM effective shots used for this term: `392`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.6894865525672371e-02`, `x_rem=3.6797598336794864e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.4778325123152709e-02`, `x_rem=-2.0219728239698743e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.5175879396984924e-02`, `x_rem=-4.8127694855296221e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-3.9443155452436193e-02`, `x_rem=-5.3966188828461131e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=7.9069767441860464e-02`, `x_rem=1.0818338318635708e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=9.4339622641509441e-02`, `x_r_target=1.2907562366851696e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 37

- pauli term from int row: `(-2.3610110167205777e-02)*Y(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*Z(q(4))*Z(q(5))`
- int observable row: `[2, 3, 2, 3, 3, 3]`
- Hamiltonian weight w_37: `-2.3610110167205777e-02`
- OGM effective shots used for this term: `392`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.4449877750611247e-03`, `x_rem=-3.7054012100530033e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-6.4039408866995079e-02`, `x_rem=-9.7052306570699839e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.5175879396984924e-02`, `x_rem=5.3309365147647655e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=6.9605568445475635e-03`, `x_rem=1.0548787203561706e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=4.6511627906976744e-02`, `x_rem=7.0488795112172015e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.8867924528301886e-02`, `x_r_target=-2.8594511224749014e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 38

- pauli term from int row: `(-2.6709685222962787e-02)*Y(q(0))*Z(q(1))*Y(q(2))*Z(q(4))`
- int observable row: `[2, 3, 2, 0, 3, 0]`
- Hamiltonian weight w_38: `-2.6709685222962787e-02`
- OGM effective shots used for this term: `392`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=3.1784841075794622e-02`, `x_rem=4.0496091493263969e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=7.8817733990147784e-02`, `x_rem=1.0041925832963973e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.0100502512562814e-02`, `x_rem=-2.5609433971001069e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.6241299303944315e-02`, `x_rem=2.0692541485849480e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-3.7209302325581395e-02`, `x_rem=-4.7407231257946186e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-6.1320754716981132e-02`, `x_r_target=-7.8126893494079919e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 39

- pauli term from int row: `(3.2798138198784431e-02)*Y(q(0))*X(q(2))*X(q(3))*Y(q(5))`
- int observable row: `[2, 0, 1, 1, 0, 2]`
- Hamiltonian weight w_39: `3.2798138198784431e-02`
- OGM effective shots used for this term: `271`
- fitted unmit coeffs: `a_u=2.0591743434819998e+00`, `b_u=3.0957076534000001e-02`
- fitted rem coeffs: `a_r=1.5603589984160000e+00`, `b_r=3.0957076534000001e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.8604651162790697e-01`, `x_rem=-2.4552183428768568e-01`, `y_exact=-4.5855193109831927e-01`
  - train[1] t_remaining=2: `x_unmit=-2.5868725868725867e-01`, `x_rem=-3.4138436514614984e-01`, `y_exact=-4.5855193109831927e-01`
  - train[2] t_remaining=2: `x_unmit=-2.1285140562248997e-01`, `x_rem=-2.8089571302290944e-01`, `y_exact=-4.5855193109831927e-01`
  - train[3] t_remaining=2: `x_unmit=-2.8205128205128205e-01`, `x_rem=-3.7221739621177991e-01`, `y_exact=-4.5855193109831927e-01`
  - train[4] t_remaining=2: `x_unmit=-2.5170068027210885e-01`, `x_rem=-3.3216431832628224e-01`, `y_exact=-4.5855203824909219e-01`
- target x values: `x_u_target=-2.7407407407407408e-01`, `x_r_target=-3.6169003551084067e-01`
- target contribution to E_cdr_unmit: `-1.5039650597470627e-02`
- target contribution to E_cdr_rem: `-1.5039650597470627e-02`

### term 40

- pauli term from int row: `(-3.2798138198784431e-02)*Y(q(0))*X(q(2))*Y(q(3))*X(q(5))`
- int observable row: `[2, 0, 1, 2, 0, 1]`
- Hamiltonian weight w_40: `-3.2798138198784431e-02`
- OGM effective shots used for this term: `297`
- fitted unmit coeffs: `a_u=2.3234635574990001e+00`, `b_u=-2.0929505189000001e-02`
- fitted rem coeffs: `a_r=1.7606266710300000e+00`, `b_r=-2.0929505189000001e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.2344322344322345e-01`, `x_rem=2.9487352167426722e-01`, `y_exact=4.5855193109831927e-01`
  - train[1] t_remaining=2: `x_unmit=2.9209621993127149e-01`, `x_rem=3.8547349841988449e-01`, `y_exact=4.5855193109831927e-01`
  - train[2] t_remaining=2: `x_unmit=3.3108108108108109e-01`, `x_rem=4.3692103449675151e-01`, `y_exact=4.5855193109831927e-01`
  - train[3] t_remaining=2: `x_unmit=1.8571428571428572e-01`, `x_rem=2.4508340244074342e-01`, `y_exact=4.5855193109831927e-01`
  - train[4] t_remaining=2: `x_unmit=1.6862745098039217e-01`, `x_rem=2.2253425078329941e-01`, `y_exact=4.5389205225939810e-01`
- target x values: `x_u_target=3.3858267716535434e-01`, `x_r_target=4.4682073976174291e-01`
- target contribution to E_cdr_unmit: `-1.5064770801205560e-02`
- target contribution to E_cdr_rem: `-1.5064770801205562e-02`

### term 41

- pauli term from int row: `(1.1507931652465444e-02)*Y(q(0))*Y(q(2))*X(q(3))*X(q(4))`
- int observable row: `[2, 0, 2, 1, 1, 0]`
- Hamiltonian weight w_41: `1.1507931652465444e-02`
- OGM effective shots used for this term: `123`
- fitted unmit coeffs: `a_u=1.4729734094730000e+00`, `b_u=-3.7295581920000001e-02`
- fitted rem coeffs: `a_r=1.1337155502639999e+00`, `b_r=-3.7295581920000001e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=7.8740157480314960e-03`, `x_rem=1.0230269682658385e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.4084507042253521e-02`, `x_rem=1.8299214784473449e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.2727272727272726e-01`, `x_rem=-1.6535835905242324e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.6393442622950818e-01`, `x_rem=2.1299086060616582e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-5.7142857142857141e-02`, `x_rem=-7.4242528554149242e-02`, `y_exact=-1.2637056219494841e-01`
- target x values: `x_u_target=-6.1946902654867256e-02`, `x_r_target=-8.0484157060913941e-02`
- target contribution to E_cdr_unmit: `-4.0242677701799557e-04`
- target contribution to E_cdr_rem: `-4.0242677701799573e-04`

### term 42

- pauli term from int row: `(1.1507931652465444e-02)*Y(q(0))*Y(q(2))*Y(q(3))*Y(q(4))`
- int observable row: `[2, 0, 2, 2, 2, 0]`
- Hamiltonian weight w_42: `1.1507931652465444e-02`
- OGM effective shots used for this term: `141`
- fitted unmit coeffs: `a_u=1.9780552115659999e+00`, `b_u=6.8702224692999997e-02`
- fitted rem coeffs: `a_r=1.5224660120880000e+00`, `b_r=6.8702224692999997e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.8951048951048952e-02`, `x_rem=6.3599368866316605e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.5447154471544716e-01`, `x_rem=2.0069626621345216e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.7777777777777776e-02`, `x_rem=-3.6090118047155846e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.3600000000000001e-01`, `x_rem=1.7669721795887514e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-2.2900763358778626e-02`, `x_rem=-2.9753685107578903e-02`, `y_exact=-1.4220230542299955e-01`
- target x values: `x_u_target=-1.0937500000000000e-01`, `x_r_target=-1.4210483981067623e-01`
- target contribution to E_cdr_unmit: `-1.0781615623990737e-03`
- target contribution to E_cdr_rem: `-1.0781615623990733e-03`

### term 43

- pauli term from int row: `(9.2962528164499229e-03)*Y(q(0))*Y(q(2))*Z(q(3))*Z(q(4))`
- int observable row: `[2, 0, 2, 3, 3, 0]`
- Hamiltonian weight w_43: `9.2962528164499229e-03`
- OGM effective shots used for this term: `591`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.4876033057851240e-02`, `x_rem=1.9327600408724856e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-8.6805555555555552e-02`, `x_rem=-1.1278161889736205e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.3651877133105802e-02`, `x_rem=-1.7737122862765943e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=2.3474178403755867e-02`, `x_rem=3.0498691307455725e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.2944983818770227e-02`, `x_rem=1.6818695788965894e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-3.8461538461538464e-02`, `x_r_target=-4.9970932680677316e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 44

- pauli term from int row: `(-4.8770671106108651e-02)*Z(q(0))`
- int observable row: `[3, 0, 0, 0, 0, 0]`
- Hamiltonian weight w_44: `-4.8770671106108651e-02`
- OGM effective shots used for this term: `3965`
- fitted unmit coeffs: `a_u=1.4732125202290001e+00`, `b_u=1.2377437590000000e-02`
- fitted rem coeffs: `a_r=1.4013197492420000e+00`, `b_r=1.2377437590000000e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-6.0583580613254207e-01`, `x_rem=-6.3691737398290815e-01`, `y_exact=-8.0212781855891135e-01`
  - train[1] t_remaining=2: `x_unmit=-6.3302752293577980e-01`, `x_rem=-6.6550412419657279e-01`, `y_exact=-8.0212781855891135e-01`
  - train[2] t_remaining=2: `x_unmit=-6.2105263157894741e-01`, `x_rem=-6.5291487760612643e-01`, `y_exact=-8.0212781855891135e-01`
  - train[3] t_remaining=2: `x_unmit=-6.2216624685138544e-01`, `x_rem=-6.5408562536941284e-01`, `y_exact=-8.0212781855891135e-01`
  - train[4] t_remaining=2: `x_unmit=-6.9275222730556230e-01`, `x_rem=-7.2829292189398909e-01`, `y_exact=-8.7963615534371042e-01`
- target x values: `x_u_target=-6.1066666666666669e-01`, `x_r_target=-6.4199607513316526e-01`
- target contribution to E_cdr_unmit: `3.8709475108004980e-02`
- target contribution to E_cdr_rem: `3.8709475108004973e-02`

### term 45

- pauli term from int row: `(3.2811891784718882e-03)*Z(q(0))*X(q(1))*X(q(2))*Z(q(4))`
- int observable row: `[3, 1, 1, 0, 3, 0]`
- Hamiltonian weight w_45: `3.2811891784718882e-03`
- OGM effective shots used for this term: `61`
- fitted unmit coeffs: `a_u=-1.1788800000000000e-07`, `b_u=3.8460000000000002e-09`
- fitted rem coeffs: `a_r=-9.2528999999999994e-08`, `b_r=3.8460000000000002e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-4.8780487804878050e-02`, `x_rem=-6.2149723905234350e-02`, `y_exact=-2.4375673923771046e-08`
  - train[1] t_remaining=2: `x_unmit=1.4084507042253521e-02`, `x_rem=1.7944638592356423e-02`, `y_exact=-2.4375673923771046e-08`
  - train[2] t_remaining=2: `x_unmit=-2.2033898305084745e-01`, `x_rem=-2.8072714272449067e-01`, `y_exact=-2.4375673923771046e-08`
  - train[3] t_remaining=2: `x_unmit=-1.4925373134328358e-02`, `x_rem=-1.9015960299362717e-02`, `y_exact=-2.4375673923771046e-08`
  - train[4] t_remaining=2: `x_unmit=-1.4705882352941177e-01`, `x_rem=-1.8736313824372108e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=0.0000000000000000e+00`, `x_r_target=5.6934514083341356e-17`
- target contribution to E_cdr_unmit: `-7.5102200279597459e-11`
- target contribution to E_cdr_rem: `-7.5102200279597498e-11`

### term 46

- pauli term from int row: `(3.2811891784718882e-03)*Z(q(0))*Y(q(1))*Y(q(2))*Z(q(4))`
- int observable row: `[3, 2, 2, 0, 3, 0]`
- Hamiltonian weight w_46: `3.2811891784718882e-03`
- OGM effective shots used for this term: `64`
- fitted unmit coeffs: `a_u=8.2824506918200003e-01`, `b_u=-9.0910771168999996e-02`
- fitted rem coeffs: `a_r=6.5007848720799999e-01`, `b_r=-9.0910771168999996e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-4.6153846153846156e-02`, `x_rem=-5.8803200310337117e-02`, `y_exact=-1.9737926506148629e-01`
  - train[1] t_remaining=2: `x_unmit=-1.7142857142857143e-01`, `x_rem=-2.1841188686696622e-01`, `y_exact=-1.9737926506148629e-01`
  - train[2] t_remaining=2: `x_unmit=-2.0512820512820512e-01`, `x_rem=-2.6134755693483147e-01`, `y_exact=-1.9737926506148629e-01`
  - train[3] t_remaining=2: `x_unmit=-1.7647058823529413e-01`, `x_rem=-2.2483576589246529e-01`, `y_exact=-1.9737926506148629e-01`
  - train[4] t_remaining=2: `x_unmit=6.0606060606060608e-02`, `x_rem=7.7216323639836631e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.1475409836065574e-01`, `x_r_target=-1.4620467836723150e-01`
- target contribution to E_cdr_unmit: `-5.3333657364246288e-04`
- target contribution to E_cdr_rem: `-5.3333657364246277e-04`

### term 47

- pauli term from int row: `(4.7435147736005136e-02)*Z(q(0))*Z(q(1))`
- int observable row: `[3, 3, 0, 0, 0, 0]`
- Hamiltonian weight w_47: `4.7435147736005136e-02`
- OGM effective shots used for this term: `3840`
- fitted unmit coeffs: `a_u=1.6627367437839999e+00`, `b_u=9.9782954740000006e-03`
- fitted rem coeffs: `a_r=1.5018827930770000e+00`, `b_r=9.9782954740000006e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-6.2381318963305110e-01`, `x_rem=-6.9062453903951238e-01`, `y_exact=-8.8866712235826562e-01`
  - train[1] t_remaining=2: `x_unmit=-6.3412127440904420e-01`, `x_rem=-7.0203663550542417e-01`, `y_exact=-8.8866712235826562e-01`
  - train[2] t_remaining=2: `x_unmit=-6.4589665653495443e-01`, `x_rem=-7.1507317911795121e-01`, `y_exact=-8.8866712235826562e-01`
  - train[3] t_remaining=2: `x_unmit=-6.2711864406779660e-01`, `x_rem=-6.9428401271408335e-01`, `y_exact=-8.8866712235826562e-01`
  - train[4] t_remaining=2: `x_unmit=-6.4070664344364270e-01`, `x_rem=-7.0932730766418362e-01`, `y_exact=-8.7963615534371042e-01`
- target x values: `x_u_target=-6.3020572002007025e-01`, `x_r_target=-6.9770171923576352e-01`
- target contribution to E_cdr_unmit: `-4.2101803626393305e-02`
- target contribution to E_cdr_rem: `-4.2101803626393298e-02`

### term 48

- pauli term from int row: `(-2.6709685222962787e-02)*Z(q(0))*Z(q(1))*Z(q(2))*X(q(3))*Z(q(4))*X(q(5))`
- int observable row: `[3, 3, 3, 1, 3, 1]`
- Hamiltonian weight w_48: `-2.6709685222962787e-02`
- OGM effective shots used for this term: `359`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.4054054054054057e-03`, `x_rem=-8.1919410535767380e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.8867924528301886e-02`, `x_rem=2.8594511224749011e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=4.1450777202072540e-02`, `x_rem=6.2819029840899335e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.0539845758354756e-01`, `x_rem=-1.5973232105753105e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.7199017199017199e-02`, `x_rem=-2.6065266988653239e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-6.0109289617486336e-02`, `x_r_target=-9.1096175103981783e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 49

- pauli term from int row: `(-2.6709685222962787e-02)*Z(q(0))*Z(q(1))*Z(q(2))*Y(q(3))*Z(q(4))*Y(q(5))`
- int observable row: `[3, 3, 3, 2, 3, 2]`
- Hamiltonian weight w_49: `-2.6709685222962787e-02`
- OGM effective shots used for this term: `383`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.1813471502590670e-02`, `x_rem=-7.8523787301124262e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-8.3969465648854963e-02`, `x_rem=-1.2725648888571500e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=2.5316455696202532e-03`, `x_rem=3.8367318858523814e-03`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=3.5897435897435895e-02`, `x_rem=5.4402890586573757e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-6.0913705583756347e-02`, `x_rem=-9.2315274816956214e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.3017902813299233e-02`, `x_r_target=3.4883841059348511e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 50

- pauli term from int row: `(1.5264219777698207e-02)*Z(q(0))*Z(q(1))*X(q(3))*X(q(4))`
- int observable row: `[3, 3, 0, 1, 1, 0]`
- Hamiltonian weight w_50: `1.5264219777698207e-02`
- OGM effective shots used for this term: `300`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-3.3112582781456956e-02`, `x_rem=-4.2930723393239702e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-2.6666666666666668e-02`, `x_rem=-3.4573542572689048e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.7027027027027029e-02`, `x_rem=-3.5040752607455095e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=3.2786885245901639e-03`, `x_rem=4.2508453982814430e-03`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-3.1645569620253167e-02`, `x_rem=-4.1028729318855657e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=5.5865921787709494e-02`, `x_r_target=7.2430605948370891e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 51

- pauli term from int row: `(1.5264219777698207e-02)*Z(q(0))*Z(q(1))*Y(q(3))*Y(q(4))`
- int observable row: `[3, 3, 0, 2, 2, 0]`
- Hamiltonian weight w_51: `1.5264219777698207e-02`
- OGM effective shots used for this term: `283`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.5316455696202531e-02`, `x_rem=3.2822983455084553e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-7.6923076923076927e-02`, `x_rem=-9.9731372805833773e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.3434650455927049e-02`, `x_rem=-4.3348286660286388e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=-1.1921857982552016e-17`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=6.4935064935064929e-02`, `x_rem=8.4188821199729771e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=3.1250000000000000e-02`, `x_r_target=4.0515870202369977e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 52

- pauli term from int row: `(7.1866706378460071e-02)*Z(q(0))*Z(q(2))`
- int observable row: `[3, 0, 3, 0, 0, 0]`
- Hamiltonian weight w_52: `7.1866706378460071e-02`
- OGM effective shots used for this term: `3840`
- fitted unmit coeffs: `a_u=1.4854958663919999e+00`, `b_u=1.2091062253999999e-02`
- fitted rem coeffs: `a_r=1.3389622759030000e+00`, `b_r=1.2091062253999999e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-6.7616114960225815e-01`, `x_rem=-7.5015899314386969e-01`, `y_exact=-9.0261859971078762e-01`
  - train[1] t_remaining=2: `x_unmit=-6.9886947584789316e-01`, `x_rem=-7.7535247721557154e-01`, `y_exact=-9.0261859971078762e-01`
  - train[2] t_remaining=2: `x_unmit=-6.9047619047619047e-01`, `x_rem=-7.6604064599410993e-01`, `y_exact=-9.0261859971078762e-01`
  - train[3] t_remaining=2: `x_unmit=-7.0065189048239895e-01`, `x_rem=-7.7732995605825916e-01`, `y_exact=-9.0261859971078762e-01`
  - train[4] t_remaining=2: `x_unmit=-7.9049514804677778e-01`, `x_rem=-8.7700549594236066e-01`, `y_exact=-9.9999962827979516e-01`
- target x values: `x_u_target=-6.8439538384345211e-01`, `x_r_target=-7.5929436696905683e-01`
- target contribution to E_cdr_unmit: `-6.4449846908246164e-02`
- target contribution to E_cdr_rem: `-6.4449846908246150e-02`

### term 53

- pauli term from int row: `(1.4789120830937334e-02)*Z(q(0))*Z(q(2))*X(q(4))*X(q(5))`
- int observable row: `[3, 0, 3, 0, 1, 1]`
- Hamiltonian weight w_53: `1.4789120830937334e-02`
- OGM effective shots used for this term: `239`
- fitted unmit coeffs: `a_u=3.6687999999999997e-08`, `b_u=-1.9799999999999999e-10`
- fitted rem coeffs: `a_r=2.7376000000000000e-08`, `b_r=-1.9799999999999999e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.0204081632653061e-01`, `x_rem=1.3674647641785820e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=8.3333333333333332e-03`, `x_rem=1.1167628907458446e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=2.6315789473684209e-02`, `x_rem=3.5266196549868692e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=6.3291139240506333e-02`, `x_rem=8.4817434740190573e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=4.8387096774193547e-02`, `x_rem=6.4844296882016678e-02`, `y_exact=1.6878953790211426e-09`
- target x values: `x_u_target=1.1111111111111110e-01`, `x_r_target=1.4890171876611230e-01`
- target contribution to E_cdr_unmit: `4.6118904965908623e-12`
- target contribution to E_cdr_rem: `4.6118904965908687e-12`

### term 54

- pauli term from int row: `(1.4789120830937334e-02)*Z(q(0))*Z(q(2))*Y(q(4))*Y(q(5))`
- int observable row: `[3, 0, 3, 0, 2, 2]`
- Hamiltonian weight w_54: `1.4789120830937334e-02`
- OGM effective shots used for this term: `214`
- fitted unmit coeffs: `a_u=1.7311425002930001e+00`, `b_u=-1.2625867836000001e-02`
- fitted rem coeffs: `a_r=1.2917860740170000e+00`, `b_r=-1.2625867836000001e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.2970711297071130e-01`, `x_rem=1.7382250851776287e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=4.2016806722689079e-02`, `x_rem=5.6307372642647527e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.4705882352941177e-01`, `x_rem=1.9707580424926630e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-4.2918454935622317e-03`, `x_rem=-5.7515685360300852e-03`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.2048192771084338e-02`, `x_rem=1.6145969504759189e-02`, `y_exact=-6.5207182718997503e-02`
- target x values: `x_u_target=3.3333333333333333e-02`, `x_r_target=4.4670515629833700e-02`
- target contribution to E_cdr_unmit: `-2.7906217044456119e-04`
- target contribution to E_cdr_rem: `-2.7906217044456141e-04`

### term 55

- pauli term from int row: `(1.1183806637694098e-01)*Z(q(0))*Z(q(3))`
- int observable row: `[3, 0, 0, 3, 0, 0]`
- Hamiltonian weight w_55: `1.1183806637694098e-01`
- OGM effective shots used for this term: `2640`
- fitted unmit coeffs: `a_u=2.1543260000000001e-06`, `b_u=9.9999904798700001e-01`
- fitted rem coeffs: `a_r=1.9082100000000001e-06`, `b_r=9.9999904798700001e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=6.2022471910112364e-01`, `x_rem=7.0021959860830929e-01`, `y_exact=9.9999956393716483e-01`
  - train[1] t_remaining=2: `x_unmit=6.3546045089797476e-01`, `x_rem=7.1742039321507134e-01`, `y_exact=9.9999956393716483e-01`
  - train[2] t_remaining=2: `x_unmit=6.1104889884285185e-01`, `x_rem=6.8986030627397488e-01`, `y_exact=9.9999956393716483e-01`
  - train[3] t_remaining=2: `x_unmit=6.3137557959814539e-01`, `x_rem=7.1280866644275132e-01`, `y_exact=9.9999956393716483e-01`
  - train[4] t_remaining=2: `x_unmit=6.2829912023460410e-01`, `x_rem=7.0933541380652054e-01`, `y_exact=9.9999962827979516e-01`
- target x values: `x_u_target=6.0223048327137552e-01`, `x_r_target=6.7990451570056865e-01`
- target contribution to E_cdr_unmit: `1.1183801772358633e-01`
- target contribution to E_cdr_rem: `1.1183801772358637e-01`

### term 56

- pauli term from int row: `(5.3161545522654625e-02)*Z(q(0))*Z(q(4))`
- int observable row: `[3, 0, 0, 0, 3, 0]`
- Hamiltonian weight w_56: `5.3161545522654625e-02`
- OGM effective shots used for this term: `3382`
- fitted unmit coeffs: `a_u=2.2928326685779998e+00`, `b_u=4.1457105870000002e-03`
- fitted rem coeffs: `a_r=1.9999242123009999e+00`, `b_r=4.1457105870000002e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-4.6643315820198483e-01`, `x_rem=-5.3474685503381247e-01`, `y_exact=-8.0212781855891135e-01`
  - train[1] t_remaining=2: `x_unmit=-4.9155503785672683e-01`, `x_rem=-5.6354807960733178e-01`, `y_exact=-8.0212781855891135e-01`
  - train[2] t_remaining=2: `x_unmit=-4.7167630057803467e-01`, `x_rem=-5.4075790687861514e-01`, `y_exact=-8.0212781855891135e-01`
  - train[3] t_remaining=2: `x_unmit=-5.0519750519750517e-01`, `x_rem=-5.7918861968708213e-01`, `y_exact=-8.0212781855891135e-01`
  - train[4] t_remaining=2: `x_unmit=-5.3924624539529609e-01`, `x_rem=-6.1822413081759131e-01`, `y_exact=-8.8866713947199116e-01`
- target x values: `x_u_target=-4.5865970409051349e-01`, `x_r_target=-5.2583490255838650e-01`
- target contribution to E_cdr_unmit: `-4.1409929055316215e-02`
- target contribution to E_cdr_rem: `-4.1409929055316215e-02`

### term 57

- pauli term from int row: `(1.0466484457724451e-01)*Z(q(0))*Z(q(5))`
- int observable row: `[3, 0, 0, 0, 0, 3]`
- Hamiltonian weight w_57: `1.0466484457724451e-01`
- OGM effective shots used for this term: `2770`
- fitted unmit coeffs: `a_u=1.9191063921180000e+00`, `b_u=-1.7859709041000000e-02`
- fitted rem coeffs: `a_r=1.6480198713650001e+00`, `b_r=-1.7859709041000000e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-6.4265335235378029e-01`, `x_rem=-7.4836485763777016e-01`, `y_exact=-9.9999956393716483e-01`
  - train[1] t_remaining=2: `x_unmit=-6.6106647187728274e-01`, `x_rem=-7.6980679289011644e-01`, `y_exact=-9.9999956393716483e-01`
  - train[2] t_remaining=2: `x_unmit=-6.3266761768901580e-01`, `x_rem=-7.3673654686426426e-01`, `y_exact=-9.9999956393716483e-01`
  - train[3] t_remaining=2: `x_unmit=-6.5600882028665941e-01`, `x_rem=-7.6391719673577385e-01`, `y_exact=-9.9999956393716483e-01`
  - train[4] t_remaining=2: `x_unmit=-6.4693520140105076e-01`, `x_rem=-7.5335103773151813e-01`, `y_exact=-9.8983723959654712e-01`
- target x values: `x_u_target=-6.3215163215163217e-01`, `x_r_target=-7.3613568569733367e-01`
- target contribution to E_cdr_unmit: `-1.0442091973302262e-01`
- target contribution to E_cdr_rem: `-1.0442091973302260e-01`

### term 58

- pauli term from int row: `(-9.2111144935490188e-03)*X(q(1))*X(q(2))`
- int observable row: `[0, 1, 1, 0, 0, 0]`
- Hamiltonian weight w_58: `-9.2111144935490188e-03`
- OGM effective shots used for this term: `1086`
- fitted unmit coeffs: `a_u=1.7010963833240000e+00`, `b_u=1.9498838920999999e-02`
- fitted rem coeffs: `a_r=1.5307162026229999e+00`, `b_r=1.9498838920999999e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.4392523364485982e-01`, `x_rem=-1.5994519036437260e-01`, `y_exact=-1.9737926506148629e-01`
  - train[1] t_remaining=2: `x_unmit=-8.7121212121212127e-02`, `x_rem=-9.6818455698229305e-02`, `y_exact=-1.9737926506148629e-01`
  - train[2] t_remaining=2: `x_unmit=-1.3770180436847104e-01`, `x_rem=-1.5302904678674667e-01`, `y_exact=-1.9737926506148629e-01`
  - train[3] t_remaining=2: `x_unmit=-1.0818438381937912e-01`, `x_rem=-1.2022611620099968e-01`, `y_exact=-1.9737926506148629e-01`
  - train[4] t_remaining=2: `x_unmit=-9.3283582089552231e-03`, `x_rem=-1.0366674360939512e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.3931297709923665e-01`, `x_r_target=-1.5481955511253528e-01`
- target contribution to E_cdr_unmit: `2.0258411752757887e-03`
- target contribution to E_cdr_rem: `2.0258411752757887e-03`

### term 59

- pauli term from int row: `(1.0106830583060904e-02)*X(q(1))*X(q(2))*X(q(3))*X(q(5))`
- int observable row: `[0, 1, 1, 1, 0, 1]`
- Hamiltonian weight w_59: `1.0106830583060904e-02`
- OGM effective shots used for this term: `126`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.2820512820512819e-01`, `x_rem=-1.6947479669748269e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.2698412698412698e-01`, `x_rem=-1.6786075101464959e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.3793103448275862e-01`, `x_rem=-1.8233150541246412e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=5.0000000000000003e-02`, `x_rem=6.6095170712018300e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=7.1428571428571425e-02`, `x_rem=9.4421672445740401e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=5.9829059829059832e-02`, `x_r_target=7.9088238458825261e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 60

- pauli term from int row: `(1.0106830583060904e-02)*X(q(1))*X(q(2))*Y(q(3))*Y(q(5))`
- int observable row: `[0, 1, 1, 2, 0, 2]`
- Hamiltonian weight w_60: `1.0106830583060904e-02`
- OGM effective shots used for this term: `139`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.0134228187919462e-02`, `x_rem=-2.6615504984705378e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.2068965517241380e-01`, `x_rem=-1.5954006723590616e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.1428571428571428e-01`, `x_rem=1.5107467591318460e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-3.3333333333333333e-02`, `x_rem=-4.4063447141345490e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.2977099236641221e-01`, `x_rem=1.7154471787852077e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.7006802721088435e-01`, `x_r_target=2.2481350582319135e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 61

- pauli term from int row: `(1.4789120830937334e-02)*X(q(1))*X(q(2))*Z(q(3))*Z(q(4))`
- int observable row: `[0, 1, 1, 3, 3, 0]`
- Hamiltonian weight w_61: `1.4789120830937334e-02`
- OGM effective shots used for this term: `253`
- fitted unmit coeffs: `a_u=-3.3259000000000000e-08`, `b_u=5.7699999999999997e-09`
- fitted rem coeffs: `a_r=-2.5556000000000001e-08`, `b_r=5.7699999999999997e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.7619047619047616e-02`, `x_rem=6.1973017727382347e-02`, `y_exact=-2.4375673923771046e-08`
  - train[1] t_remaining=2: `x_unmit=-7.5757575757575760e-02`, `x_rem=-9.8593437293562833e-02`, `y_exact=-2.4375673923771046e-08`
  - train[2] t_remaining=2: `x_unmit=-5.5555555555555552e-02`, `x_rem=-7.2301854015279415e-02`, `y_exact=-2.4375673923771046e-08`
  - train[3] t_remaining=2: `x_unmit=-2.2222222222222223e-02`, `x_rem=-2.8920741606111747e-02`, `y_exact=-2.4375673923771046e-08`
  - train[4] t_remaining=2: `x_unmit=7.9365079365079361e-03`, `x_rem=1.0328836287897051e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-6.9230769230769235e-02`, `x_r_target=-9.0099233465194342e-02`
- target contribution to E_cdr_unmit: `-3.3902690581217936e-10`
- target contribution to E_cdr_rem: `-3.3902690581217957e-10`

### term 62

- pauli term from int row: `(1.5582981492557525e-02)*X(q(1))*X(q(2))*Z(q(4))`
- int observable row: `[0, 1, 1, 0, 3, 0]`
- Hamiltonian weight w_62: `1.5582981492557525e-02`
- OGM effective shots used for this term: `821`
- fitted unmit coeffs: `a_u=2.2842692583000002e+00`, `b_u=-2.1960556704000000e-02`
- fitted rem coeffs: `a_r=1.8848742817780000e+00`, `b_r=-2.1960556704000000e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.3432835820895522e-01`, `x_rem=-1.6279183293078456e-01`, `y_exact=-1.9737926506148629e-01`
  - train[1] t_remaining=2: `x_unmit=-7.6167076167076173e-02`, `x_rem=-9.2306480206726632e-02`, `y_exact=-1.9737926506148629e-01`
  - train[2] t_remaining=2: `x_unmit=-6.3989962358845673e-02`, `x_rem=-7.7549099836120294e-02`, `y_exact=-1.9737926506148629e-01`
  - train[3] t_remaining=2: `x_unmit=-8.8699878493317133e-02`, `x_rem=-1.0749491762717260e-01`, `y_exact=-1.9737926506148629e-01`
  - train[4] t_remaining=2: `x_unmit=-5.1186017478152310e-02`, `x_rem=-6.2032066175733841e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-8.1632653061224483e-02`, `x_r_target=-9.8930184184694472e-02`
- target contribution to E_cdr_unmit: `-2.4309980951404053e-03`
- target contribution to E_cdr_rem: `-2.4309980951404044e-03`

### term 63

- pauli term from int row: `(1.1698435536077332e-02)*X(q(1))*X(q(2))*Z(q(4))*Z(q(5))`
- int observable row: `[0, 1, 1, 0, 3, 3]`
- Hamiltonian weight w_63: `1.1698435536077332e-02`
- OGM effective shots used for this term: `253`
- fitted unmit coeffs: `a_u=-8.3700000000000002e-08`, `b_u=-6.7569999999999997e-09`
- fitted rem coeffs: `a_r=-6.2352000000000001e-08`, `b_r=-6.7569999999999997e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-9.1575091575091569e-02`, `x_rem=-1.2292797218000935e-01`, `y_exact=2.4375673923771046e-08`
  - train[1] t_remaining=2: `x_unmit=1.5151515151515152e-02`, `x_rem=2.0338991760692433e-02`, `y_exact=2.4375673923771046e-08`
  - train[2] t_remaining=2: `x_unmit=1.1111111111111110e-01`, `x_rem=1.4915260624507795e-01`, `y_exact=2.4375673923771046e-08`
  - train[3] t_remaining=2: `x_unmit=1.4814814814814815e-02`, `x_rem=1.9887014166010346e-02`, `y_exact=2.4375673923771046e-08`
  - train[4] t_remaining=2: `x_unmit=-1.5873015873015872e-02`, `x_rem=-2.1307515177868247e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=3.0769230769230771e-02`, `x_r_target=4.1303798652483077e-02`
- target contribution to E_cdr_unmit: `2.3543210555524406e-10`
- target contribution to E_cdr_rem: `2.3543210555524395e-10`

### term 64

- pauli term from int row: `(-4.7092490952951533e-03)*X(q(1))*Y(q(2))*X(q(3))*Y(q(4))`
- int observable row: `[0, 1, 2, 1, 2, 0]`
- Hamiltonian weight w_64: `-4.7092490952951533e-03`
- OGM effective shots used for this term: `96`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-7.0707070707070704e-02`, `x_rem=-9.2020541473991968e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=-3.8616453030440226e-17`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.3707865168539325e-02`, `x_rem=-4.3868540638484152e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.2676056338028169e-01`, `x_rem=1.6497042747148263e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.4285714285714285e-01`, `x_rem=1.8591905318214708e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-2.5333333333333335e-01`, `x_r_target=-3.2969645430967415e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 65

- pauli term from int row: `(4.7092490952951533e-03)*X(q(1))*Y(q(2))*Y(q(3))*X(q(4))`
- int observable row: `[0, 1, 2, 2, 1, 0]`
- Hamiltonian weight w_65: `4.7092490952951533e-03`
- OGM effective shots used for this term: `99`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.5909090909090909e-01`, `x_rem=-2.0704621831648201e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=7.5268817204301078e-02`, `x_rem=9.7957350601346305e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-7.6923076923076927e-02`, `x_rem=-1.0011025940577151e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.2048192771084338e-02`, `x_rem=1.5679920147891949e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.0344827586206896e-01`, `x_rem=-1.3463103851120994e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-2.0408163265306121e-02`, `x_r_target=-2.6559864740306707e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 66

- pauli term from int row: `(-8.0778073309737148e-03)*X(q(1))*Y(q(2))*X(q(4))*Y(q(5))`
- int observable row: `[0, 1, 2, 0, 1, 2]`
- Hamiltonian weight w_66: `-8.0778073309737148e-03`
- OGM effective shots used for this term: `99`
- fitted unmit coeffs: `a_u=6.6299999999999996e-09`, `b_u=5.2879999999999999e-09`
- fitted rem coeffs: `a_r=4.9389999999999996e-09`, `b_r=5.2879999999999999e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-4.5454545454545456e-02`, `x_rem=-6.1016975282077279e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=5.3763440860215055e-02`, `x_rem=7.2170615925037720e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=7.6923076923076927e-02`, `x_rem=1.0325949663120786e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.2048192771084338e-02`, `x_rem=1.6173174171153021e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=8.0459770114942528e-02`, `x_rem=1.0800705969471151e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-8.1632653061224483e-02`, `x_r_target=-1.0958150662903683e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 67

- pauli term from int row: `(8.0778073309737148e-03)*X(q(1))*Y(q(2))*Y(q(4))*X(q(5))`
- int observable row: `[0, 1, 2, 0, 2, 1]`
- Hamiltonian weight w_67: `8.0778073309737148e-03`
- OGM effective shots used for this term: `96`
- fitted unmit coeffs: `a_u=6.1460470177700000e-01`, `b_u=-6.4798830409999999e-03`
- fitted rem coeffs: `a_r=4.5784926611499999e-01`, `b_r=-6.4798830409999999e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.0505050505050504e-02`, `x_rem=6.7796639202308051e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=4.3478260869565216e-02`, `x_rem=5.8364063313291366e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-1.2359550561797752e-01`, `x_rem=-1.6591132604789571e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=4.2253521126760563e-02`, `x_rem=5.6720005191790157e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-9.0909090909090912e-02`, `x_rem=-1.2203395056415464e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-4.0000000000000001e-02`, `x_r_target=-5.3694938248228054e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 68

- pauli term from int row: `(4.7092490952951533e-03)*Y(q(1))*X(q(2))*X(q(3))*Y(q(4))`
- int observable row: `[0, 2, 1, 1, 2, 0]`
- Hamiltonian weight w_68: `4.7092490952951533e-03`
- OGM effective shots used for this term: `95`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.8681318681318682e-01`, `x_rem=-2.4312491569973083e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-4.9504950495049507e-02`, `x_rem=-6.4427394667080681e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-5.4945054945054944e-02`, `x_rem=-7.1507328146979660e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.8181818181818181e-02`, `x_rem=-2.3662424950455084e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=1.5384615384615385e-01`, `x_rem=2.0022051881154301e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-4.6511627906976744e-02`, `x_r_target=-6.0531784756978090e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 69

- pauli term from int row: `(-4.7092490952951533e-03)*Y(q(1))*X(q(2))*Y(q(3))*X(q(4))`
- int observable row: `[0, 2, 1, 2, 1, 0]`
- Hamiltonian weight w_69: `-4.7092490952951533e-03`
- OGM effective shots used for this term: `106`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.0869565217391304e-01`, `x_rem=-1.4146014916032928e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-3.8961038961038960e-02`, `x_rem=-5.0705196322403709e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.3255813953488372e-02`, `x_rem=-3.0265892378489104e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.1827956989247312e-01`, `x_rem=-1.5393297951640136e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-2.5333333333333335e-01`, `x_rem=-3.2969645430967404e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0526315789473684e-02`, `x_r_target=1.3699298655526639e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 70

- pauli term from int row: `(8.0778073309737148e-03)*Y(q(1))*X(q(2))*X(q(4))*Y(q(5))`
- int observable row: `[0, 2, 1, 0, 1, 2]`
- Hamiltonian weight w_70: `8.0778073309737148e-03`
- OGM effective shots used for this term: `106`
- fitted unmit coeffs: `a_u=7.1968724619400004e-01`, `b_u=1.4688658833000001e-02`
- fitted rem coeffs: `a_r=5.3613042098399999e-01`, `b_r=1.4688658833000001e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.3478260869565216e-02`, `x_rem=5.8364063313291366e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.2987012987012988e-02`, `x_rem=-1.7433421509164979e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=2.5819140107561780e-18`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.0752688172043012e-02`, `x_rem=-1.4434123185007528e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-4.0000000000000001e-02`, `x_rem=-5.3694938248228088e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=5.2631578947368418e-02`, `x_r_target=7.0651234537142102e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 71

- pauli term from int row: `(-8.0778073309737148e-03)*Y(q(1))*X(q(2))*Y(q(4))*X(q(5))`
- int observable row: `[0, 2, 1, 0, 2, 1]`
- Hamiltonian weight w_71: `-8.0778073309737148e-03`
- OGM effective shots used for this term: `95`
- fitted unmit coeffs: `a_u=-7.9700000000000004e-10`, `b_u=5.4459999999999996e-09`
- fitted rem coeffs: `a_r=-5.9300000000000002e-10`, `b_r=5.4459999999999996e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.6483516483516483e-01`, `x_rem=2.2127034992401676e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-8.9108910891089105e-02`, `x_rem=-1.1961743669159711e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-5.4945054945054944e-02`, `x_rem=-7.3756783308005586e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-5.4545454545454543e-02`, `x_rem=-7.3220370338492879e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=5.1282051282051280e-02`, `x_rem=6.8839664420805177e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-6.9767441860465115e-02`, `x_r_target=-9.3653962060862905e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 72

- pauli term from int row: `(-9.2111144935490188e-03)*Y(q(1))*Y(q(2))`
- int observable row: `[0, 2, 2, 0, 0, 0]`
- Hamiltonian weight w_72: `-9.2111144935490188e-03`
- OGM effective shots used for this term: `1069`
- fitted unmit coeffs: `a_u=-2.8548800000000000e-07`, `b_u=2.1780000000000001e-09`
- fitted rem coeffs: `a_r=-2.5689400000000002e-07`, `b_r=2.1780000000000001e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.9755409219190969e-02`, `x_rem=2.1954334262791256e-02`, `y_exact=-2.4375673923771046e-08`
  - train[1] t_remaining=2: `x_unmit=1.8832391713747645e-02`, `x_rem=2.0928577994213181e-02`, `y_exact=-2.4375673923771046e-08`
  - train[2] t_remaining=2: `x_unmit=3.3395176252319109e-02`, `x_rem=3.7112309548921987e-02`, `y_exact=-2.4375673923771046e-08`
  - train[3] t_remaining=2: `x_unmit=-1.8587360594795538e-03`, `x_rem=-2.0656273076072669e-03`, `y_exact=-2.4375673923771046e-08`
  - train[4] t_remaining=2: `x_unmit=4.7125353440150798e-03`, `x_rem=5.2370758317281923e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.2440191387559809e-02`, `x_r_target=-1.3824877884598404e-02`
- target contribution to E_cdr_unmit: `9.7209230642655411e-11`
- target contribution to E_cdr_rem: `9.7209230642655411e-11`

### term 73

- pauli term from int row: `(1.0106830583060904e-02)*Y(q(1))*Y(q(2))*X(q(3))*X(q(5))`
- int observable row: `[0, 2, 2, 1, 0, 1]`
- Hamiltonian weight w_73: `1.0106830583060904e-02`
- OGM effective shots used for this term: `123`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-3.9370078740157480e-02`, `x_rem=-5.2043441505526214e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=3.7528665621132051e-17`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.8181818181818181e-02`, `x_rem=2.4034607531643038e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=8.1967213114754092e-02`, `x_rem=1.0835273887216110e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.8571428571428571e-02`, `x_rem=3.7768668978296192e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-8.8495575221238937e-03`, `x_r_target=-1.1698260303012097e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 74

- pauli term from int row: `(1.0106830583060904e-02)*Y(q(1))*Y(q(2))*Y(q(3))*Y(q(5))`
- int observable row: `[0, 2, 2, 2, 0, 2]`
- Hamiltonian weight w_74: `1.0106830583060904e-02`
- OGM effective shots used for this term: `141`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-7.6923076923076927e-02`, `x_rem=-1.0168487801848969e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-8.1300813008130090e-03`, `x_rem=-1.0747182229596509e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.2500000000000000e-01`, `x_rem=1.6523792678004576e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-1.3600000000000001e-01`, `x_rem=-1.7977886433668969e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-5.3435114503816793e-02`, `x_rem=-7.0636060302920287e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=3.1250000000000000e-02`, `x_r_target=4.1309481695011446e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 75

- pauli term from int row: `(1.4789120830937334e-02)*Y(q(1))*Y(q(2))*Z(q(3))*Z(q(4))`
- int observable row: `[0, 2, 2, 3, 3, 0]`
- Hamiltonian weight w_75: `1.4789120830937334e-02`
- OGM effective shots used for this term: `263`
- fitted unmit coeffs: `a_u=2.0325119011799999e+00`, `b_u=-3.0586562862000000e-02`
- fitted rem coeffs: `a_r=1.5617487183590000e+00`, `b_r=-3.0586562862000000e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.2643678160919541e-01`, `x_rem=-1.6454904706925663e-01`, `y_exact=-1.9737926506148629e-01`
  - train[1] t_remaining=2: `x_unmit=-1.4999999999999999e-01`, `x_rem=-1.9521500584125442e-01`, `y_exact=-1.9737926506148629e-01`
  - train[2] t_remaining=2: `x_unmit=-2.5563909774436089e-01`, `x_rem=-3.3269725306278952e-01`, `y_exact=-1.9737926506148629e-01`
  - train[3] t_remaining=2: `x_unmit=-2.1739130434782608e-02`, `x_rem=-2.8292029832065857e-02`, `y_exact=-1.9737926506148629e-01`
  - train[4] t_remaining=2: `x_unmit=3.9370078740157480e-02`, `x_rem=5.1237534341536592e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-5.7471264367816091e-02`, `x_r_target=-7.4795021395116637e-02`
- target contribution to E_cdr_unmit: `-1.9789233486797110e-03`
- target contribution to E_cdr_rem: `-1.9789233486797110e-03`

### term 76

- pauli term from int row: `(1.5582981492557525e-02)*Y(q(1))*Y(q(2))*Z(q(4))`
- int observable row: `[0, 2, 2, 0, 3, 0]`
- Hamiltonian weight w_76: `1.5582981492557525e-02`
- OGM effective shots used for this term: `805`
- fitted unmit coeffs: `a_u=-7.5222999999999998e-08`, `b_u=5.5050000000000003e-09`
- fitted rem coeffs: `a_r=-6.2071000000000000e-08`, `b_r=5.5050000000000003e-09`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-6.3051702395964691e-03`, `x_rem=-7.6412027507093876e-03`, `y_exact=-2.4375673923771046e-08`
  - train[1] t_remaining=2: `x_unmit=8.7829360100376407e-03`, `x_rem=1.0643994095153760e-02`, `y_exact=-2.4375673923771046e-08`
  - train[2] t_remaining=2: `x_unmit=5.5825242718446605e-02`, `x_rem=6.7654318917567152e-02`, `y_exact=-2.4375673923771046e-08`
  - train[3] t_remaining=2: `x_unmit=-8.4439083232810616e-03`, `x_rem=-1.0233128219345662e-02`, `y_exact=-2.4375673923771046e-08`
  - train[4] t_remaining=2: `x_unmit=1.2658227848101266e-02`, `x_rem=1.5340439952689954e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.2437810945273632e-02`, `x_r_target=-1.5073317863961547e-02`
- target contribution to E_cdr_unmit: `-3.0442297467422664e-10`
- target contribution to E_cdr_rem: `-3.0442297467422659e-10`

### term 77

- pauli term from int row: `(1.1698435536077332e-02)*Y(q(1))*Y(q(2))*Z(q(4))*Z(q(5))`
- int observable row: `[0, 2, 2, 0, 3, 3]`
- Hamiltonian weight w_77: `1.1698435536077332e-02`
- OGM effective shots used for this term: `263`
- fitted unmit coeffs: `a_u=1.9611982009569999e+00`, `b_u=6.1968823779000003e-02`
- fitted rem coeffs: `a_r=1.4609929836519999e+00`, `b_r=6.1968823779000003e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=7.2796934865900387e-02`, `x_rem=9.7720673057120011e-02`, `y_exact=1.9737926506148629e-01`
  - train[1] t_remaining=2: `x_unmit=1.3333333333333333e-01`, `x_rem=1.7898312749409345e-01`, `y_exact=1.9737926506148629e-01`
  - train[2] t_remaining=2: `x_unmit=1.8045112781954886e-01`, `x_rem=2.4223280412734466e-01`, `y_exact=1.9737926506148629e-01`
  - train[3] t_remaining=2: `x_unmit=7.9710144927536225e-02`, `x_rem=1.0700078274103413e-01`, `y_exact=1.9737926506148629e-01`
  - train[4] t_remaining=2: `x_unmit=7.8740157480314960e-03`, `x_rem=1.0569869733903144e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=8.0459770114942528e-02`, `x_r_target=1.0800705969471157e-01`
- target contribution to E_cdr_unmit: `1.6783390352011463e-03`
- target contribution to E_cdr_rem: `1.6783390352011450e-03`

### term 78

- pauli term from int row: `(-1.2754595390400281e-01)*Z(q(1))`
- int observable row: `[0, 3, 0, 0, 0, 0]`
- Hamiltonian weight w_78: `-1.2754595390400281e-01`
- OGM effective shots used for this term: `4993`
- fitted unmit coeffs: `a_u=1.5960033352670000e+00`, `b_u=-2.5158246850000001e-03`
- fitted rem coeffs: `a_r=1.5155647671700001e+00`, `b_r=-2.5158246850000001e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=6.8602576808721505e-01`, `x_rem=7.2243657127971295e-01`, `y_exact=9.0261859971078806e-01`
  - train[1] t_remaining=2: `x_unmit=6.8189038919777600e-01`, `x_rem=7.1808170724281417e-01`, `y_exact=9.0261859971078806e-01`
  - train[2] t_remaining=2: `x_unmit=6.8561348490787921e-01`, `x_rem=7.2200240617931699e-01`, `y_exact=9.0261859971078806e-01`
  - train[3] t_remaining=2: `x_unmit=6.6713119649611785e-01`, `x_rem=7.0253917069936622e-01`, `y_exact=9.0261859971078806e-01`
  - train[4] t_remaining=2: `x_unmit=7.6358695652173914e-01`, `x_rem=8.0411431815684442e-01`, `y_exact=9.9999962827979516e-01`
- target x values: `x_u_target=6.9060559006211175e-01`, `x_r_target=7.2725946720946932e-01`
- target contribution to E_cdr_unmit: `-1.1671774762944807e-01`
- target contribution to E_cdr_rem: `-1.1671774762944807e-01`

### term 79

- pauli term from int row: `(5.1610885078092486e-02)*Z(q(1))*Z(q(2))`
- int observable row: `[0, 3, 3, 0, 0, 0]`
- Hamiltonian weight w_79: `5.1610885078092486e-02`
- OGM effective shots used for this term: `3840`
- fitted unmit coeffs: `a_u=1.6861988031340001e+00`, `b_u=-1.9886263891000001e-02`
- fitted rem coeffs: `a_r=1.5173107497630001e+00`, `b_r=-1.9886263891000001e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.4426481909160895e-01`, `x_rem=6.0484557081243451e-01`, `y_exact=8.0212781855891180e-01`
  - train[1] t_remaining=2: `x_unmit=5.6320657759506676e-01`, `x_rem=6.2589568893937331e-01`, `y_exact=8.0212781855891180e-01`
  - train[2] t_remaining=2: `x_unmit=5.7244174265450865e-01`, `x_rem=6.3615879705510281e-01`, `y_exact=8.0212781855891180e-01`
  - train[3] t_remaining=2: `x_unmit=5.6453715775749669e-01`, `x_rem=6.2737437264191320e-01`, `y_exact=8.0212781855891180e-01`
  - train[4] t_remaining=2: `x_unmit=6.3224682756904704e-01`, `x_rem=7.0262063594998747e-01`, `y_exact=8.7963615534371042e-01`
- target x values: `x_u_target=5.7350727546412439e-01`, `x_r_target=6.3734293164886013e-01`
- target contribution to E_cdr_unmit: `4.2105407825486205e-02`
- target contribution to E_cdr_rem: `4.2105407825486205e-02`

### term 80

- pauli term from int row: `(1.2439498134905486e-02)*Z(q(1))*Z(q(2))*X(q(3))*X(q(4))`
- int observable row: `[0, 3, 3, 1, 1, 0]`
- Hamiltonian weight w_80: `1.2439498134905486e-02`
- OGM effective shots used for this term: `300`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=3.3112582781456956e-02`, `x_rem=4.3093820274007613e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=2.6666666666666668e-02`, `x_rem=3.4704889927334129e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=6.0810810810810814e-02`, `x_rem=7.9141218584292350e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=4.2622950819672129e-02`, `x_rem=5.5470930621558640e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=3.1645569620253167e-02`, `x_rem=4.1184600388450308e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0614525139664804e-01`, `x_r_target=-1.3814097247612045e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 81

- pauli term from int row: `(-1.6982826390582344e-02)*Z(q(1))*Z(q(2))*X(q(3))*Z(q(4))*X(q(5))`
- int observable row: `[0, 3, 3, 1, 3, 1]`
- Hamiltonian weight w_81: `-1.6982826390582344e-02`
- OGM effective shots used for this term: `359`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.0810810810810811e-02`, `x_rem=-1.5584348660324373e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=0.0000000000000000e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.0725388601036270e-02`, `x_rem=-2.9876730592331736e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.0539845758354756e-01`, `x_rem=1.5193738378992352e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-4.1769041769041768e-02`, `x_rem=-6.0212256187616962e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=5.4644808743169399e-03`, `x_r_target=7.8773347053552253e-03`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 82

- pauli term from int row: `(9.2962528164499229e-03)*Z(q(1))*Z(q(2))*X(q(3))*X(q(5))`
- int observable row: `[0, 3, 3, 1, 0, 1]`
- Hamiltonian weight w_82: `9.2962528164499229e-03`
- OGM effective shots used for this term: `598`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.4634146341463415e-02`, `x_rem=1.9344928013273657e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.2048192771084338e-02`, `x_rem=1.5926547159522471e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.2573289902280132e-03`, `x_rem=-4.3058743134865235e-03`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=3.1948881789137379e-02`, `x_rem=4.2233335918222550e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-5.0381679389312976e-02`, `x_rem=-6.6599713999896296e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-4.8062015503875968e-02`, `x_r_target=-6.3533342389847036e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 83

- pauli term from int row: `(1.2439498134905486e-02)*Z(q(1))*Z(q(2))*Y(q(3))*Y(q(4))`
- int observable row: `[0, 3, 3, 2, 2, 0]`
- Hamiltonian weight w_83: `1.2439498134905486e-02`
- OGM effective shots used for this term: `283`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.2658227848101267e-01`, `x_rem=-1.6473840155380121e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=4.3478260869565216e-02`, `x_rem=5.6584059664131721e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.0395136778115501e-03`, `x_rem=3.9557245357903642e-03`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=4.0268456375838924e-02`, `x_rem=5.2406712977517970e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=6.4935064935064939e-03`, `x_rem=8.4508660537339717e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-8.1250000000000003e-02`, `x_r_target=-1.0574146149734616e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 84

- pauli term from int row: `(-1.6982826390582344e-02)*Z(q(1))*Z(q(2))*Y(q(3))*Z(q(4))*Y(q(5))`
- int observable row: `[0, 3, 3, 2, 3, 2]`
- Hamiltonian weight w_84: `-1.6982826390582344e-02`
- OGM effective shots used for this term: `383`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.2435233160621761e-01`, `x_rem=1.7926038355399043e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=5.8524173027989825e-02`, `x_rem=8.4365653371094515e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.7974683544303799e-02`, `x_rem=5.4742490547342017e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-6.1538461538461542e-02`, `x_rem=-8.8710907758769594e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=6.5989847715736044e-02`, `x_rem=9.5127813523046084e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-3.3248081841432228e-02`, `x_r_target=-4.7928847222608893e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 85

- pauli term from int row: `(9.2962528164499229e-03)*Z(q(1))*Z(q(2))*Y(q(3))*Y(q(5))`
- int observable row: `[0, 3, 3, 2, 0, 2]`
- Hamiltonian weight w_85: `9.2962528164499229e-03`
- OGM effective shots used for this term: `597`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.2800000000000000e-02`, `x_rem=6.9796500271891312e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=4.2789223454833596e-02`, `x_rem=5.6563220577638462e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-7.4962518740629685e-03`, `x_rem=-9.9093209463295730e-03`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-4.6548956661316206e-02`, `x_rem=-6.1533224739920712e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.3328149300155521e-02`, `x_rem=3.0837560207784571e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-7.4484944532488120e-02`, `x_r_target=-9.8461902487000305e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 86

- pauli term from int row: `(-9.2111144935490188e-03)*Z(q(1))*Z(q(2))*X(q(4))*X(q(5))`
- int observable row: `[0, 3, 3, 0, 1, 1]`
- Hamiltonian weight w_86: `-9.2111144935490188e-03`
- OGM effective shots used for this term: `239`
- fitted unmit coeffs: `a_u=1.7859223943300000e+00`, `b_u=4.4847637721999997e-02`
- fitted rem coeffs: `a_r=1.3304214159440000e+00`, `b_r=4.4847637721999997e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-2.8571428571428571e-02`, `x_rem=-3.8353527320162908e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-8.3333333333333332e-03`, `x_rem=-1.1186445468380843e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.5087719298245612e-02`, `x_rem=-4.7100823024761464e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-7.1729957805907171e-02`, `x_rem=-9.6288391373404766e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-6.4516129032258063e-02`, `x_rem=-8.6604739110045265e-02`, `y_exact=-6.5207182718997503e-02`
- target x values: `x_u_target=-8.2437275985663083e-02`, `x_r_target=-1.1066161108505783e-01`
- target contribution to E_cdr_unmit: `3.2372700930540145e-04`
- target contribution to E_cdr_rem: `3.2372700930540139e-04`

### term 87

- pauli term from int row: `(-9.2111144935490188e-03)*Z(q(1))*Z(q(2))*Y(q(4))*Y(q(5))`
- int observable row: `[0, 3, 3, 0, 2, 2]`
- Hamiltonian weight w_87: `-9.2111144935490188e-03`
- OGM effective shots used for this term: `214`
- fitted unmit coeffs: `a_u=-1.1316000000000000e-08`, `b_u=2.4900000000000002e-10`
- fitted rem coeffs: `a_r=-8.4300000000000000e-09`, `b_r=2.4900000000000002e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-9.6234309623430964e-02`, `x_rem=-1.2918238281477465e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.3445378151260504e-01`, `x_rem=-1.8048718738900194e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.3529411764705882e-01`, `x_rem=-3.1585257793075333e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-5.5793991416309016e-02`, `x_rem=-7.4896373093021995e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-6.0240963855421686e-02`, `x_rem=-8.0865870855765160e-02`, `y_exact=1.6878953790211426e-09`
- target x values: `x_u_target=-7.4999999999999997e-02`, `x_r_target=-1.0067800921542763e-01`
- target contribution to E_cdr_unmit: `-4.7758592285235699e-12`
- target contribution to E_cdr_rem: `-4.7758592285235723e-12`

### term 88

- pauli term from int row: `(4.2983656068340519e-03)*Z(q(1))*X(q(3))*X(q(4))`
- int observable row: `[0, 3, 0, 1, 1, 0]`
- Hamiltonian weight w_88: `4.2983656068340519e-03`
- OGM effective shots used for this term: `300`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.9867549668874173e-02`, `x_rem=2.4501422454989766e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=2.6666666666666668e-02`, `x_rem=3.2886353695141826e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=8.7837837837837843e-02`, `x_rem=1.0832498261068672e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-5.5737704918032788e-02`, `x_rem=-6.8737870428370207e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-3.1645569620253167e-02`, `x_rem=-3.9026527328095514e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-7.2625698324022353e-02`, `x_r_target=-8.9564790091517535e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 89

- pauli term from int row: `(2.3326675518445836e-03)*Z(q(1))*X(q(3))*X(q(4))*Z(q(5))`
- int observable row: `[0, 3, 0, 1, 1, 3]`
- Hamiltonian weight w_89: `2.3326675518445836e-03`
- OGM effective shots used for this term: `61`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.2280701754385964e-01`, `x_rem=1.6775621740112104e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=3.3333333333333333e-02`, `x_rem=4.5533830437447152e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=5.8823529411764705e-02`, `x_rem=8.0353818419024406e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=5.8823529411764705e-02`, `x_rem=8.0353818419024337e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=0.0000000000000000e+00`, `x_rem=-2.9388256534195321e-17`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-2.1518987341772153e-01`, `x_r_target=-2.9395257624174742e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 90

- pauli term from int row: `(-2.3610110167205774e-02)*Z(q(1))*X(q(3))*Z(q(4))*X(q(5))`
- int observable row: `[0, 3, 0, 1, 3, 1]`
- Hamiltonian weight w_90: `-2.3610110167205774e-02`
- OGM effective shots used for this term: `359`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.1621621621621623e-02`, `x_rem=2.9535457581046792e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.8867924528301886e-02`, `x_rem=2.5773866285347453e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.6269430051813469e-02`, `x_rem=4.9544582341263715e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=2.8277634961439587e-02`, `x_rem=3.8627671065186525e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.4570024570024569e-03`, `x_rem=3.3563019978462023e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0928961748633879e-01`, `x_r_target=1.4929124733589227e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 91

- pauli term from int row: `(4.2983656068340519e-03)*Z(q(1))*Y(q(3))*Y(q(4))`
- int observable row: `[0, 3, 0, 2, 2, 0]`
- Hamiltonian weight w_91: `4.2983656068340519e-03`
- OGM effective shots used for this term: `283`
- fitted unmit coeffs: `a_u=-0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=-0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.2658227848101267e-01`, `x_rem=-1.5610610931238211e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=1.0033444816053512e-02`, `x_rem=1.2373628062553358e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.1276595744680851e-02`, `x_rem=-2.6239111990804646e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=4.0268456375838924e-02`, `x_rem=4.9660601217496041e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=5.8441558441558440e-02`, `x_rem=7.2072366052664691e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-3.1250000000000000e-02`, `x_r_target=-3.8538695736494322e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 92

- pauli term from int row: `(2.3326675518445836e-03)*Z(q(1))*Y(q(3))*Y(q(4))*Z(q(5))`
- int observable row: `[0, 3, 0, 2, 2, 3]`
- Hamiltonian weight w_92: `2.3326675518445836e-03`
- OGM effective shots used for this term: `69`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-3.8961038961038960e-02`, `x_rem=-5.3221360251561635e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.1475409836065574e-01`, `x_rem=-1.5675580970268690e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=5.2631578947368418e-02`, `x_rem=7.1895521743337562e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=4.6153846153846156e-02`, `x_rem=6.3046842144157580e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=5.0847457627118647e-02`, `x_rem=6.9458385413055018e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.5000000000000001e-02`, `x_r_target=3.4150372828085362e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 93

- pauli term from int row: `(-2.3610110167205774e-02)*Z(q(1))*Y(q(3))*Z(q(4))*Y(q(5))`
- int observable row: `[0, 3, 0, 2, 3, 2]`
- Hamiltonian weight w_93: `-2.3610110167205774e-02`
- OGM effective shots used for this term: `383`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=9.3264248704663211e-02`, `x_rem=1.2740035459182103e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=7.8880407124681931e-02`, `x_rem=1.0775181248556190e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=1.2658227848101266e-02`, `x_rem=1.7291328014220446e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-2.5641025641025640e-02`, `x_rem=-3.5026023413420877e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=3.0456852791878174e-02`, `x_rem=4.1604515120510083e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-5.8823529411764705e-02`, `x_r_target=-8.0353818419024350e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 94

- pauli term from int row: `(5.3161545522654625e-02)*Z(q(1))*Z(q(3))`
- int observable row: `[0, 3, 0, 3, 0, 0]`
- Hamiltonian weight w_94: `5.3161545522654625e-02`
- OGM effective shots used for this term: `3668`
- fitted unmit coeffs: `a_u=2.2411567791599998e+00`, `b_u=1.2447420270000000e-02`
- fitted rem coeffs: `a_r=1.9817821470390000e+00`, `b_r=1.2447420270000000e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.1947698174884227e-01`, `x_rem=-5.8746586298775572e-01`, `y_exact=-8.8866712235826562e-01`
  - train[1] t_remaining=2: `x_unmit=-5.3756906077348066e-01`, `x_rem=-6.0792582404641637e-01`, `y_exact=-8.8866712235826562e-01`
  - train[2] t_remaining=2: `x_unmit=-5.2867965367965364e-01`, `x_rem=-5.9787297590626642e-01`, `y_exact=-8.8866712235826562e-01`
  - train[3] t_remaining=2: `x_unmit=-5.5561658884921716e-01`, `x_rem=-6.2833540335080695e-01`, `y_exact=-8.8866712235826562e-01`
  - train[4] t_remaining=2: `x_unmit=-5.4279581432787760e-01`, `x_rem=-6.1383665242830354e-01`, `y_exact=-8.7963615534371042e-01`
- target x values: `x_u_target=-5.5125100887812750e-01`, `x_r_target=-6.2339845850962372e-01`
- target contribution to E_cdr_unmit: `-4.7092276668010490e-02`
- target contribution to E_cdr_rem: `-4.7092276668010503e-02`

### term 95

- pauli term from int row: `(8.1752954675092818e-02)*Z(q(1))*Z(q(4))`
- int observable row: `[0, 3, 0, 0, 3, 0]`
- Hamiltonian weight w_95: `8.1752954675092818e-02`
- OGM effective shots used for this term: `4063`
- fitted unmit coeffs: `a_u=2.3042828093310002e+00`, `b_u=-6.4518980890000003e-03`
- fitted rem coeffs: `a_r=2.0065307584140002e+00`, `b_r=-6.4518980890000003e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.5744888023369032e-01`, `x_rem=6.4016953959802003e-01`, `y_exact=9.0261859971078806e-01`
  - train[1] t_remaining=2: `x_unmit=5.1883353584447145e-01`, `x_rem=5.9582400745038688e-01`, `y_exact=9.0261859971078806e-01`
  - train[2] t_remaining=2: `x_unmit=5.5080213903743314e-01`, `x_rem=6.3253647869806573e-01`, `y_exact=9.0261859971078806e-01`
  - train[3] t_remaining=2: `x_unmit=5.3955773955773956e-01`, `x_rem=6.1962350623868245e-01`, `y_exact=9.0261859971078806e-01`
  - train[4] t_remaining=2: `x_unmit=5.9458174904942962e-01`, `x_rem=6.8281260944105249e-01`, `y_exact=9.8983723959654712e-01`
- target x values: `x_u_target=5.4385117178062337e-01`, `x_r_target=6.2455404718490626e-01`
- target contribution to E_cdr_unmit: `7.4400137954699325e-02`
- target contribution to E_cdr_rem: `7.4400137954699352e-02`

### term 96

- pauli term from int row: `(5.9688692409066199e-02)*Z(q(1))*Z(q(5))`
- int observable row: `[0, 3, 0, 0, 0, 3]`
- Hamiltonian weight w_96: `5.9688692409066199e-02`
- OGM effective shots used for this term: `3451`
- fitted unmit coeffs: `a_u=2.0417909699529999e+00`, `b_u=-1.0531587798000000e-02`
- fitted rem coeffs: `a_r=1.7504251117350000e+00`, `b_r=-1.0531587798000000e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.4216867469879515e-01`, `x_rem=6.3241500408678408e-01`, `y_exact=8.8866712235826562e-01`
  - train[1] t_remaining=2: `x_unmit=5.8057911670078965e-01`, `x_rem=6.7721903089479152e-01`, `y_exact=8.8866712235826562e-01`
  - train[2] t_remaining=2: `x_unmit=5.6043956043956045e-01`, `x_rem=6.5372715807066106e-01`, `y_exact=8.8866712235826562e-01`
  - train[3] t_remaining=2: `x_unmit=5.6308411214953269e-01`, `x_rem=6.5681190689241664e-01`, `y_exact=8.8866712235826562e-01`
  - train[4] t_remaining=2: `x_unmit=5.7555178268251272e-01`, `x_rem=6.7135487530616278e-01`, `y_exact=8.8866713947199116e-01`
- target x values: `x_u_target=5.9672408924032760e-01`, `x_r_target=6.9605140419678146e-01`
- target contribution to E_cdr_unmit: `5.3043379136692083e-02`
- target contribution to E_cdr_rem: `5.3043379136692041e-02`

### term 97

- pauli term from int row: `(-2.2856223074292265e-01)*Z(q(2))`
- int observable row: `[0, 0, 3, 0, 0, 0]`
- Hamiltonian weight w_97: `-2.2856223074292265e-01`
- OGM effective shots used for this term: `4488`
- fitted unmit coeffs: `a_u=1.2487780663470001e+00`, `b_u=-8.9737612620000001e-03`
- fitted rem coeffs: `a_r=1.1833420956709999e+00`, `b_r=-8.9737612620000001e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=7.4807311164941648e-01`, `x_rem=7.8943975480098827e-01`, `y_exact=8.8866712235826606e-01`
  - train[1] t_remaining=2: `x_unmit=7.7534486533829650e-01`, `x_rem=8.1821957085088270e-01`, `y_exact=8.8866712235826606e-01`
  - train[2] t_remaining=2: `x_unmit=7.5410913872452334e-01`, `x_rem=7.9580955965019362e-01`, `y_exact=8.8866712235826606e-01`
  - train[3] t_remaining=2: `x_unmit=7.7639751552795033e-01`, `x_rem=8.1933043006326522e-01`, `y_exact=8.8866712235826606e-01`
  - train[4] t_remaining=2: `x_unmit=7.5352571056628337e-01`, `x_rem=7.9519386931857694e-01`, `y_exact=8.7963615534371042e-01`
- target x values: `x_u_target=7.4448569556671762e-01`, `x_r_target=7.8565396324052095e-01`
- target contribution to E_cdr_unmit: `-2.0231035569267120e-01`
- target contribution to E_cdr_rem: `-2.0231035569267120e-01`

### term 98

- pauli term from int row: `(4.5870037211547705e-03)*Z(q(2))*X(q(3))*Z(q(4))*X(q(5))`
- int observable row: `[0, 0, 3, 1, 3, 1]`
- Hamiltonian weight w_98: `4.5870037211547705e-03`
- OGM effective shots used for this term: `359`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=3.7837837837837840e-02`, `x_rem=5.1796141207454127e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-2.3584905660377360e-02`, `x_rem=-3.2285330604376730e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-3.6269430051813469e-02`, `x_rem=-4.9649150898336859e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=8.4832904884318772e-02`, `x_rem=1.1612759532556279e-01`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-2.2113022113022112e-02`, `x_rem=-3.0270472134226449e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.6393442622950821e-02`, `x_r_target=2.2440951108615957e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 99

- pauli term from int row: `(4.5870037211547705e-03)*Z(q(2))*Y(q(3))*Z(q(4))*Y(q(5))`
- int observable row: `[0, 0, 3, 2, 3, 2]`
- Hamiltonian weight w_99: `4.5870037211547705e-03`
- OGM effective shots used for this term: `383`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=4.1450777202072540e-02`, `x_rem=5.6741886740956440e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=2.7989821882951654e-02`, `x_rem=3.8315211689265417e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=2.2784810126582278e-02`, `x_rem=3.1190081414253595e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=4.1025641025641026e-02`, `x_rem=5.6159918671818396e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=4.5685279187817257e-02`, `x_rem=6.2538488114873933e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.7902813299232736e-02`, `x_r_target=-2.4507125635240445e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 100

- pauli term from int row: `(1.0466484457724451e-01)*Z(q(2))*Z(q(3))`
- int observable row: `[0, 0, 3, 3, 0, 0]`
- Hamiltonian weight w_100: `1.0466484457724451e-01`
- OGM effective shots used for this term: `2744`
- fitted unmit coeffs: `a_u=1.9023957567180001e+00`, `b_u=4.2719235790000002e-03`
- fitted rem coeffs: `a_r=1.6786837559940000e+00`, `b_r=4.2719235790000002e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-6.0741548527808076e-01`, `x_rem=-6.8836350958398862e-01`, `y_exact=-9.0261859971078762e-01`
  - train[1] t_remaining=2: `x_unmit=-6.0415122312824310e-01`, `x_rem=-6.8466423124136366e-01`, `y_exact=-9.0261859971078762e-01`
  - train[2] t_remaining=2: `x_unmit=-5.7796182931220741e-01`, `x_rem=-6.5498467338019006e-01`, `y_exact=-9.0261859971078762e-01`
  - train[3] t_remaining=2: `x_unmit=-6.1280714817572601e-01`, `x_rem=-6.9447369953579385e-01`, `y_exact=-9.0261859971078762e-01`
  - train[4] t_remaining=2: `x_unmit=-6.6176997491938372e-01`, `x_rem=-7.4996162184483228e-01`, `y_exact=-9.9999962827979516e-01`
- target x values: `x_u_target=-5.9463379260333571e-01`, `x_r_target=-6.7387844780789108e-01`
- target contribution to E_cdr_unmit: `-9.4068807313510389e-02`
- target contribution to E_cdr_rem: `-9.4068807313510361e-02`

### term 101

- pauli term from int row: `(3.2811891784718882e-03)*Z(q(2))*Z(q(3))*X(q(4))*X(q(5))`
- int observable row: `[0, 0, 3, 3, 1, 1]`
- Hamiltonian weight w_101: `3.2811891784718882e-03`
- OGM effective shots used for this term: `118`
- fitted unmit coeffs: `a_u=2.0771999999999999e-08`, `b_u=5.2199999999999996e-10`
- fitted rem coeffs: `a_r=1.5174000000000001e-08`, `b_r=5.2199999999999996e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-8.0645161290322578e-02`, `x_rem=-1.1039500142141723e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-5.1724137931034482e-02`, `x_rem=-7.0805069877184823e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-4.2016806722689079e-02`, `x_rem=-5.7516723429645963e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-3.8461538461538464e-02`, `x_rem=-5.2649923754829783e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=2.0879120879120880e-01`, `x_rem=2.8581387181193296e-01`, `y_exact=1.6878953790211426e-09`
- target x values: `x_u_target=9.0909090909090912e-02`, `x_r_target=1.2444527432959761e-01`
- target contribution to E_cdr_unmit: `3.0084803065598851e-12`
- target contribution to E_cdr_rem: `3.0084803065598851e-12`

### term 102

- pauli term from int row: `(3.2811891784718882e-03)*Z(q(2))*Z(q(3))*Y(q(4))*Y(q(5))`
- int observable row: `[0, 0, 3, 3, 2, 2]`
- Hamiltonian weight w_102: `3.2811891784718882e-03`
- OGM effective shots used for this term: `111`
- fitted unmit coeffs: `a_u=9.7124582976700002e-01`, `b_u=1.4695102045000000e-02`
- fitted rem coeffs: `a_r=7.0950926749900001e-01`, `b_r=1.4695102045000000e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.1538461538461539e-01`, `x_rem=-1.5794977126448922e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=9.4339622641509441e-02`, `x_rem=1.2914132241750692e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.4482758620689655e-02`, `x_rem=4.7203379918123241e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=7.7519379844961239e-03`, `x_rem=1.0611612539733129e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-1.1320754716981132e-01`, `x_rem=-1.5496958690100834e-01`, `y_exact=-6.5207182718997503e-02`
- target x values: `x_u_target=-1.6666666666666666e-01`, `x_r_target=-2.2814966960426231e-01`
- target contribution to E_cdr_unmit: `-1.2974667085285523e-04`
- target contribution to E_cdr_rem: `-1.2974667085285528e-04`

### term 103

- pauli term from int row: `(1.5582981492557527e-02)*Z(q(2))*X(q(4))*X(q(5))`
- int observable row: `[0, 0, 3, 0, 1, 1]`
- Hamiltonian weight w_103: `1.5582981492557527e-02`
- OGM effective shots used for this term: `357`
- fitted unmit coeffs: `a_u=1.7595710437380001e+00`, `b_u=3.1305946926000000e-02`
- fitted rem coeffs: `a_r=1.3803612037010000e+00`, `b_r=3.1305946926000000e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-1.3550135501355014e-02`, `x_rem=-1.7272599376869018e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-1.1235955056179775e-02`, `x_rem=-1.4322672292280158e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=2.5936599423631124e-02`, `x_rem=3.3061845838952178e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-8.7976539589442824e-03`, `x_rem=-1.1214526398940767e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-7.9646017699115043e-02`, `x_rem=-1.0152619916917173e-01`, `y_exact=-6.5207182718997503e-02`
- target x values: `x_u_target=-1.4285714285714285e-01`, `x_r_target=-1.8210254771613343e-01`
- target contribution to E_cdr_unmit: `-1.5509629026113979e-03`
- target contribution to E_cdr_rem: `-1.5509629026113983e-03`

### term 104

- pauli term from int row: `(1.5582981492557527e-02)*Z(q(2))*Y(q(4))*Y(q(5))`
- int observable row: `[0, 0, 3, 0, 2, 2]`
- Hamiltonian weight w_104: `1.5582981492557527e-02`
- OGM effective shots used for this term: `325`
- fitted unmit coeffs: `a_u=-6.7510000000000004e-09`, `b_u=3.8200000000000003e-10`
- fitted rem coeffs: `a_r=-5.2959999999999999e-09`, `b_r=3.8200000000000003e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=2.6239067055393587e-02`, `x_rem=3.3447406723371445e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-2.3255813953488372e-02`, `x_rem=-2.9644600790998489e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-9.7938144329896906e-02`, `x_rem=-1.2484349920745229e-01`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-3.8674033149171269e-02`, `x_rem=-4.9298479768456013e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-3.0985915492957747e-02`, `x_rem=-3.9498299082090911e-02`, `y_exact=1.6878953790211426e-09`
- target x values: `x_u_target=5.1724137931034482e-02`, `x_r_target=6.5933681069634514e-02`
- target contribution to E_cdr_unmit: `5.8093600142029054e-12`
- target contribution to E_cdr_rem: `5.8093600142029102e-12`

### term 105

- pauli term from int row: `(5.9688692409066199e-02)*Z(q(2))*Z(q(4))`
- int observable row: `[0, 0, 3, 0, 3, 0]`
- Hamiltonian weight w_105: `5.9688692409066199e-02`
- OGM effective shots used for this term: `3676`
- fitted unmit coeffs: `a_u=1.9339579238969999e+00`, `b_u=-3.3860033400000001e-04`
- fitted rem coeffs: `a_r=1.6805111908040000e+00`, `b_r=-3.3860033400000001e-04`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.7726657645466850e-01`, `x_rem=6.6432718558768289e-01`, `y_exact=8.8866712235826606e-01`
  - train[1] t_remaining=2: `x_unmit=5.9369994660971703e-01`, `x_rem=6.8323895874432850e-01`, `y_exact=8.8866712235826606e-01`
  - train[2] t_remaining=2: `x_unmit=5.7115852011882262e-01`, `x_rem=6.5729794114411155e-01`, `y_exact=8.8866712235826606e-01`
  - train[3] t_remaining=2: `x_unmit=6.0185185185185186e-01`, `x_rem=6.9262029569950923e-01`, `y_exact=8.8866712235826606e-01`
  - train[4] t_remaining=2: `x_unmit=5.7444561774023228e-01`, `x_rem=6.6108078324973718e-01`, `y_exact=8.8866713947199116e-01`
- target x values: `x_u_target=5.7282079047103407e-01`, `x_r_target=6.5921090723259135e-01`
- target contribution to E_cdr_unmit: `5.3043378868424147e-02`
- target contribution to E_cdr_rem: `5.3043378868424140e-02`

### term 106

- pauli term from int row: `(1.0340487754605730e-01)*Z(q(2))*Z(q(5))`
- int observable row: `[0, 0, 3, 0, 0, 3]`
- Hamiltonian weight w_106: `1.0340487754605730e-01`
- OGM effective shots used for this term: `2645`
- fitted unmit coeffs: `a_u=1.7085692510199999e+00`, `b_u=-2.7891639070000002e-03`
- fitted rem coeffs: `a_r=1.4616695126620001e+00`, `b_r=-2.7891639070000002e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=6.3041023710952204e-01`, `x_rem=7.3689677271284004e-01`, `y_exact=9.0261859971078762e-01`
  - train[1] t_remaining=2: `x_unmit=6.3573353869849825e-01`, `x_rem=7.4311926646401993e-01`, `y_exact=9.0261859971078762e-01`
  - train[2] t_remaining=2: `x_unmit=6.2204724409448819e-01`, `x_rem=7.2712113424723612e-01`, `y_exact=9.0261859971078762e-01`
  - train[3] t_remaining=2: `x_unmit=6.3573085846867750e-01`, `x_rem=7.4311613349981853e-01`, `y_exact=9.0261859971078762e-01`
  - train[4] t_remaining=2: `x_unmit=6.8614479970599052e-01`, `x_rem=8.0204580882972043e-01`, `y_exact=9.8983723959654712e-01`
- target x values: `x_u_target=6.0738007380073800e-01`, `x_r_target=7.0997644049376762e-01`
- target contribution to E_cdr_unmit: `8.9753861408928742e-02`
- target contribution to E_cdr_rem: `8.9753861408928728e-02`

### term 107

- pauli term from int row: `(-3.8063028881711678e-03)*X(q(3))*X(q(4))`
- int observable row: `[0, 0, 0, 1, 1, 0]`
- Hamiltonian weight w_107: `-3.8063028881711678e-03`
- OGM effective shots used for this term: `549`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.1282051282051280e-02`, `x_rem=-6.0055541286358961e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=3.1690140845070422e-02`, `x_rem=3.7111786956887327e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=6.8965517241379309e-02`, `x_rem=8.0764348626482754e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-2.0109689213893969e-02`, `x_rem=-2.3550116281396690e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-3.3557046979865771e-03`, `x_rem=-3.9298089096778607e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.0408163265306121e-02`, `x_r_target=2.3899654185387746e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 108

- pauli term from int row: `(-3.8063028881711678e-03)*Y(q(3))*Y(q(4))`
- int observable row: `[0, 0, 0, 2, 2, 0]`
- Hamiltonian weight w_108: `-3.8063028881711678e-03`
- OGM effective shots used for this term: `563`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=-0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=-0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-9.8684210526315784e-03`, `x_rem=-1.1556740675171059e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-6.6914498141263934e-02`, `x_rem=-7.8362434912683993e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=-2.4469820554649267e-02`, `x_rem=-2.8656192212495924e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=1.2891344383057090e-02`, `x_rem=1.5096834964250457e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-3.5087719298245615e-03`, `x_rem=-4.1090633511719358e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-5.5462184873949577e-02`, `x_r_target=-6.4950824903818483e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 109

- pauli term from int row: `(-4.8770671106108679e-02)*Z(q(3))`
- int observable row: `[0, 0, 0, 3, 0, 0]`
- Hamiltonian weight w_109: `-4.8770671106108679e-02`
- OGM effective shots used for this term: `4413`
- fitted unmit coeffs: `a_u=1.6610033977200001e+00`, `b_u=2.2571537224000002e-02`
- fitted rem coeffs: `a_r=1.5467263639569999e+00`, `b_r=2.2571537224000002e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.8222422738551771e-01`, `x_rem=-6.2524079401365729e-01`, `y_exact=-8.0212781855891135e-01`
  - train[1] t_remaining=2: `x_unmit=-5.7616198803497465e-01`, `x_rem=-6.1873065725405363e-01`, `y_exact=-8.0212781855891135e-01`
  - train[2] t_remaining=2: `x_unmit=-5.9496516071027195e-01`, `x_rem=-6.3892306777305841e-01`, `y_exact=-8.0212781855891135e-01`
  - train[3] t_remaining=2: `x_unmit=-5.8914027149321269e-01`, `x_rem=-6.3266781732518551e-01`, `y_exact=-8.0212781855891135e-01`
  - train[4] t_remaining=2: `x_unmit=-6.4153498871331827e-01`, `x_rem=-6.8893362190004104e-01`, `y_exact=-8.7963615534371042e-01`
- target x values: `x_u_target=-5.9910011248593931e-01`, `x_r_target=-6.4336352285861187e-01`
- target contribution to E_cdr_unmit: `4.0019959654399445e-02`
- target contribution to E_cdr_rem: `4.0019959654399459e-02`

### term 110

- pauli term from int row: `(4.7435147736005136e-02)*Z(q(3))*Z(q(4))`
- int observable row: `[0, 0, 0, 3, 3, 0]`
- Hamiltonian weight w_110: `4.7435147736005136e-02`
- OGM effective shots used for this term: `3837`
- fitted unmit coeffs: `a_u=1.7850817041180000e+00`, `b_u=-2.5372162590000000e-03`
- fitted rem coeffs: `a_r=1.5242998319959999e+00`, `b_r=-2.5372162590000000e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-5.5429747812660835e-01`, `x_rem=-6.4912838410986518e-01`, `y_exact=-8.0212781855891135e-01`
  - train[1] t_remaining=2: `x_unmit=-5.2498684902682802e-01`, `x_rem=-6.1480320303726033e-01`, `y_exact=-8.0212781855891135e-01`
  - train[2] t_remaining=2: `x_unmit=-5.4297585042846019e-01`, `x_rem=-6.3586981775659401e-01`, `y_exact=-8.0212781855891135e-01`
  - train[3] t_remaining=2: `x_unmit=-5.3713838936669278e-01`, `x_rem=-6.2903366602244559e-01`, `y_exact=-8.0212781855891135e-01`
  - train[4] t_remaining=2: `x_unmit=-6.0695118834653716e-01`, `x_rem=-7.1079025193572698e-01`, `y_exact=-8.8866713947199116e-01`
- target x values: `x_u_target=-5.3779144248014343e-01`, `x_r_target=-6.2979844545767760e-01`
- target contribution to E_cdr_unmit: `-3.8027625172409506e-02`
- target contribution to E_cdr_rem: `-3.8027625172409520e-02`

### term 111

- pauli term from int row: `(7.1866706378460071e-02)*Z(q(3))*Z(q(5))`
- int observable row: `[0, 0, 0, 3, 0, 3]`
- Hamiltonian weight w_111: `7.1866706378460071e-02`
- OGM effective shots used for this term: `3837`
- fitted unmit coeffs: `a_u=1.6778113933030001e+00`, `b_u=6.6154940949999996e-03`
- fitted rem coeffs: `a_r=1.4105148308140001e+00`, `b_r=6.6154940949999996e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-7.1641791044776115e-01`, `x_rem=-8.5218113716823485e-01`, `y_exact=-9.9999956393716483e-01`
  - train[1] t_remaining=2: `x_unmit=-7.1699105733824298e-01`, `x_rem=-8.5286289702065099e-01`, `y_exact=-9.9999956393716483e-01`
  - train[2] t_remaining=2: `x_unmit=-7.0916645027265646e-01`, `x_rem=-8.4355550471539920e-01`, `y_exact=-9.9999956393716483e-01`
  - train[3] t_remaining=2: `x_unmit=-7.1488141777430281e-01`, `x_rem=-8.5035347477366963e-01`, `y_exact=-9.9999956393716483e-01`
  - train[4] t_remaining=2: `x_unmit=-6.9690774341937134e-01`, `x_rem=-8.2897373813182040e-01`, `y_exact=-9.8983723959654712e-01`
- target x values: `x_u_target=-6.8741993338457597e-01`, `x_r_target=-8.1768796117569331e-01`
- target contribution to E_cdr_unmit: `-7.0872929144169622e-02`
- target contribution to E_cdr_rem: `-7.0872929144169594e-02`

### term 112

- pauli term from int row: `(1.1698435536077332e-02)*X(q(4))*X(q(5))`
- int observable row: `[0, 0, 0, 0, 1, 1]`
- Hamiltonian weight w_112: `1.1698435536077332e-02`
- OGM effective shots used for this term: `606`
- fitted unmit coeffs: `a_u=5.0568000000000000e-08`, `b_u=-2.3700000000000001e-10`
- fitted rem coeffs: `a_r=4.1863000000000002e-08`, `b_r=-2.3700000000000001e-10`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=1.7944535073409460e-02`, `x_rem=2.1675609811773609e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-2.5641025641025640e-02`, `x_rem=-3.0972374859247590e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=4.7120418848167540e-02`, `x_rem=5.6917819767517815e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-5.6603773584905662e-02`, `x_rem=-6.8372978462867315e-02`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=-2.7463651050080775e-02`, `x_rem=-3.3173965317740146e-02`, `y_exact=-1.6878953790211426e-09`
- target x values: `x_u_target=-1.3157894736842105e-02`, `x_r_target=-1.5893718677771795e-02`
- target contribution to E_cdr_unmit: `-4.1781348002150448e-12`
- target contribution to E_cdr_rem: `-4.1781348002150496e-12`

### term 113

- pauli term from int row: `(1.1698435536077332e-02)*Y(q(4))*Y(q(5))`
- int observable row: `[0, 0, 0, 0, 2, 2]`
- Hamiltonian weight w_113: `1.1698435536077332e-02`
- OGM effective shots used for this term: `605`
- fitted unmit coeffs: `a_u=1.7702014282959999e+00`, `b_u=3.4531235248000003e-02`
- fitted rem coeffs: `a_r=1.4654924079600000e+00`, `b_r=3.4531235248000003e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=-3.9370078740157480e-02`, `x_rem=-4.7556008642151835e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=2: `x_unmit=-3.9451114922813037e-02`, `x_rem=-4.7653894080180274e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=2: `x_unmit=3.8690476190476192e-02`, `x_rem=4.6735101350114654e-02`, `y_exact=0.0000000000000000e+00`
  - train[3] t_remaining=2: `x_unmit=-4.9423393739703456e-03`, `x_rem=-5.9699635231169204e-03`, `y_exact=0.0000000000000000e+00`
  - train[4] t_remaining=2: `x_unmit=4.7001620745542948e-02`, `x_rem=5.6774320852202645e-02`, `y_exact=6.5207182718997503e-02`
- target x values: `x_u_target=4.0128410914927769e-02`, `x_r_target=4.8472015229159560e-02`
- target contribution to E_cdr_unmit: `3.5929326979896507e-04`
- target contribution to E_cdr_rem: `3.5929326979896524e-04`

### term 114

- pauli term from int row: `(-1.2754595390400281e-01)*Z(q(4))`
- int observable row: `[0, 0, 0, 0, 3, 0]`
- Hamiltonian weight w_114: `-1.2754595390400281e-01`
- OGM effective shots used for this term: `6108`
- fitted unmit coeffs: `a_u=1.6095505576500000e+00`, `b_u=2.4908369180000000e-03`
- fitted rem coeffs: `a_r=1.4759578613650000e+00`, `b_r=2.4908369180000000e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=7.4219898709361221e-01`, `x_rem=8.0937730326457169e-01`, `y_exact=9.9999956393716483e-01`
  - train[1] t_remaining=2: `x_unmit=7.3458488428548308e-01`, `x_rem=8.0107402866464916e-01`, `y_exact=9.9999956393716483e-01`
  - train[2] t_remaining=2: `x_unmit=7.3605887162714634e-01`, `x_rem=8.0268143034585215e-01`, `y_exact=9.9999956393716483e-01`
  - train[3] t_remaining=2: `x_unmit=7.4845829276209019e-01`, `x_rem=8.1620315459333725e-01`, `y_exact=9.9999956393716483e-01`
  - train[4] t_remaining=2: `x_unmit=7.3869509043927650e-01`, `x_rem=8.0555626002102143e-01`, `y_exact=9.8983723959654712e-01`
- target x values: `x_u_target=7.2877474235236384e-01`, `x_r_target=7.9473799602220729e-01`
- target contribution to E_cdr_unmit: `-1.2713233009492134e-01`
- target contribution to E_cdr_rem: `-1.2713233009492134e-01`

### term 115

- pauli term from int row: `(5.1610885078092486e-02)*Z(q(4))*Z(q(5))`
- int observable row: `[0, 0, 0, 0, 3, 3]`
- Hamiltonian weight w_115: `5.1610885078092486e-02`
- OGM effective shots used for this term: `3837`
- fitted unmit coeffs: `a_u=1.9301605423079999e+00`, `b_u=-8.3141140680000006e-03`
- fitted rem coeffs: `a_r=1.5979173757750000e+00`, `b_r=-8.3141140680000006e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=5.3113741636644363e-01`, `x_rem=6.4157289929747652e-01`, `y_exact=8.0212781855891135e-01`
  - train[1] t_remaining=2: `x_unmit=4.9763282482903737e-01`, `x_rem=6.0110194532197836e-01`, `y_exact=8.0212781855891135e-01`
  - train[2] t_remaining=2: `x_unmit=5.1285380420669957e-01`, `x_rem=6.1948771060336183e-01`, `y_exact=8.0212781855891135e-01`
  - train[3] t_remaining=2: `x_unmit=5.1159760229345841e-01`, `x_rem=6.1797031589768514e-01`, `y_exact=8.0212781855891135e-01`
  - train[4] t_remaining=2: `x_unmit=5.6861742908254531e-01`, `x_rem=6.8684585443680313e-01`, `y_exact=8.7963615534371042e-01`
- target x values: `x_u_target=5.0704586215731484e-01`, `x_r_target=6.1247216602910282e-01`
- target contribution to E_cdr_unmit: `4.1181362471464762e-02`
- target contribution to E_cdr_rem: `4.1181362471464783e-02`

### term 116

- pauli term from int row: `(-2.2856223074292262e-01)*Z(q(5))`
- int observable row: `[0, 0, 0, 0, 0, 3]`
- Hamiltonian weight w_116: `-2.2856223074292262e-01`
- OGM effective shots used for this term: `3967`
- fitted unmit coeffs: `a_u=1.4159517453620001e+00`, `b_u=-1.8527273480000001e-03`
- fitted rem coeffs: `a_r=1.2783212357130000e+00`, `b_r=-1.8527273480000001e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=2: `x_unmit=6.4179104477611937e-01`, `x_rem=7.1088950462574163e-01`, `y_exact=8.0212781855891135e-01`
  - train[1] t_remaining=2: `x_unmit=6.3395360693346925e-01`, `x_rem=7.0220824870787468e-01`, `y_exact=8.0212781855891135e-01`
  - train[2] t_remaining=2: `x_unmit=6.3682092555331993e-01`, `x_rem=7.0538427730762054e-01`, `y_exact=8.0212781855891135e-01`
  - train[3] t_remaining=2: `x_unmit=6.3375314861460952e-01`, `x_rem=7.0198620803567746e-01`, `y_exact=8.0212781855891135e-01`
  - train[4] t_remaining=2: `x_unmit=7.1485148514851482e-01`, `x_rem=7.9181600038603772e-01`, `y_exact=8.8866713947199116e-01`
- target x values: `x_u_target=6.4992614475627775e-01`, `x_r_target=7.1990047048768036e-01`
- target contribution to E_cdr_unmit: `-1.8671379928358356e-01`
- target contribution to E_cdr_rem: `-1.8671379928358353e-01`

## Complete expanded energy expressions (all summing terms)

### E_cdr_unmit

`E_cdr_unmit = (-7.1586579073350247e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-3.6212695980335891e-03) + (-4.1025047692791559e-03) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-4.5118710136187763e-04) + (-1.4596047865139635e-03) + (0.0000000000000000e+00) + (-1.3881182339500770e-02) + (-1.3712372165613327e-02) + (-1.6096278174941763e-03) + (-1.0415374662528039e-03) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-1.5039650597470627e-02) + (-1.5064770801205560e-02) + (-4.0242677701799557e-04) + (-1.0781615623990737e-03) + (0.0000000000000000e+00) + (3.8709475108004980e-02) + (-7.5102200279597459e-11) + (-5.3333657364246288e-04) + (-4.2101803626393305e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-6.4449846908246164e-02) + (4.6118904965908623e-12) + (-2.7906217044456119e-04) + (1.1183801772358633e-01) + (-4.1409929055316215e-02) + (-1.0442091973302262e-01) + (2.0258411752757887e-03) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-3.3902690581217936e-10) + (-2.4309980951404053e-03) + (2.3543210555524406e-10) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (9.7209230642655411e-11) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-1.9789233486797110e-03) + (-3.0442297467422664e-10) + (1.6783390352011463e-03) + (-1.1671774762944807e-01) + (4.2105407825486205e-02) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (3.2372700930540145e-04) + (-4.7758592285235699e-12) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-4.7092276668010490e-02) + (7.4400137954699325e-02) + (5.3043379136692083e-02) + (-2.0231035569267120e-01) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-9.4068807313510389e-02) + (3.0084803065598851e-12) + (-1.2974667085285523e-04) + (-1.5509629026113979e-03) + (5.8093600142029054e-12) + (5.3043378868424147e-02) + (8.9753861408928742e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (4.0019959654399445e-02) + (-3.8027625172409506e-02) + (-7.0872929144169622e-02) + (-4.1781348002150448e-12) + (3.5929326979896507e-04) + (-1.2713233009492134e-01) + (4.1181362471464762e-02) + (-1.8671379928358356e-01)`

### E_cdr_rem

`E_cdr_rem = (-7.1586579073350247e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-3.6212695980335865e-03) + (-4.1025047692791567e-03) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-4.5118710136187774e-04) + (-1.4596047865139629e-03) + (0.0000000000000000e+00) + (-1.3881182339500770e-02) + (-1.3712372165613324e-02) + (-1.6096278174941745e-03) + (-1.0415374662528039e-03) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-1.5039650597470627e-02) + (-1.5064770801205562e-02) + (-4.0242677701799573e-04) + (-1.0781615623990733e-03) + (0.0000000000000000e+00) + (3.8709475108004973e-02) + (-7.5102200279597498e-11) + (-5.3333657364246277e-04) + (-4.2101803626393298e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-6.4449846908246150e-02) + (4.6118904965908687e-12) + (-2.7906217044456141e-04) + (1.1183801772358637e-01) + (-4.1409929055316215e-02) + (-1.0442091973302260e-01) + (2.0258411752757887e-03) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-3.3902690581217957e-10) + (-2.4309980951404044e-03) + (2.3543210555524395e-10) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (9.7209230642655411e-11) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-1.9789233486797110e-03) + (-3.0442297467422659e-10) + (1.6783390352011450e-03) + (-1.1671774762944807e-01) + (4.2105407825486205e-02) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (3.2372700930540139e-04) + (-4.7758592285235723e-12) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-4.7092276668010503e-02) + (7.4400137954699352e-02) + (5.3043379136692041e-02) + (-2.0231035569267120e-01) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-9.4068807313510361e-02) + (3.0084803065598851e-12) + (-1.2974667085285528e-04) + (-1.5509629026113983e-03) + (5.8093600142029102e-12) + (5.3043378868424140e-02) + (8.9753861408928728e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (4.0019959654399459e-02) + (-3.8027625172409520e-02) + (-7.0872929144169594e-02) + (-4.1781348002150496e-12) + (3.5929326979896524e-04) + (-1.2713233009492134e-01) + (4.1181362471464783e-02) + (-1.8671379928358353e-01)`

## Headline values (target circuit)

- raw finite-shot (unmit / REM): `-7.608572787115 / -7.650660660605 Eh`
- cdr corrected (unmit / REM): `-7.843478792710 / -7.843478792710 Eh`
- reference exact noiseless: `-7.829052399882 Eh`
- Energy error (exact - cdr_rem): `0.014426392827 Eh`

