# full CDR details for LiH testing case

## Configuration used
- bond_length: `2.2`
- target params [theta1, theta2, theta3]: `[0.0, 0.07, 0.0]`
- measurement_scheme: `ogm`
- num_shots: `8192`
- sampling_seed: `1234`
- apply_readout_noise: `True`
- apply_rem in CDR training/target estimation: `True`
- readout p_0_success: `[0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514]`
- readout p_1_success: `[0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514]`
- base_noise_cfg: `{'two_qubit_depol_prob': 0.015, 'one_qubit_depol_prob': 0.0015}`
- cdr_cfg: `{"num_circuits": 3, "t_max": 1, "min_snap_fraction": 0.25, "seed": 42, "cdr_fit_scope": "per_pauli"}`
- training resolvers count: `3`
- training non-Clifford counts (t_remaining): `[0, 1, 1]`
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
- Hamiltonian identity term: `(-7.2476524580835413) * I`
- offset: `-7.2476524580835413`
- E_cdr_unmit: `-7.8083317864674022`
- E_cdr_rem: `-7.8083317864674022`
- delta (unmit - rem): `0.0000000000000000e+00`

## Int encoding map
- `I=0, X=1, Y=2, Z=3`

## Per-term full details (Pauli term, training pairs, fits, shots, target contribution)
### term 0
- pauli term from int row: `(1.1735799722042688e-02)*Z(q(0))`
- int observable row: `[3, 0, 0, 0, 0, 0]`
- Hamiltonian weight w_0: `1.1735799722042688e-02`
- OGM effective shots used for this term: `4857`
- fitted unmit coeffs: `a_u=2.1819594651539462e-01`, `b_u=-8.4420386632536260e-01`
- fitted rem coeffs: `a_r=2.0754798432544094e-01`, `b_r=-8.4420386632536415e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-7.0877335499593830e-01`, `x_rem=-7.4513599137504016e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-7.0245209148980015e-01`, `x_rem=-7.3849042419028621e-01`, `y_exact=-9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=-7.0838496489054115e-01`, `x_rem=-7.4472767545262963e-01`, `y_exact=-9.9755099946510994e-01`
- target x values: `x_u_target=-7.1428571428571430e-01`, `x_r_target=-7.5093115463174365e-01`
- target contribution to E_cdr_unmit: `-1.1736481734387273e-02`
- target contribution to E_cdr_rem: `-1.1736481734387270e-02`

### term 1
- pauli term from int row: `(2.1322111528987456e-02)*Y(q(0))*Y(q(1))*Z(q(3))`
- int observable row: `[2, 2, 0, 3, 0, 0]`
- Hamiltonian weight w_1: `2.1322111528987456e-02`
- OGM effective shots used for this term: `561`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-6.6162570888468802e-02`, `x_rem=-7.8660528039020877e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=6.8100358422939072e-02`, `x_rem=8.0964359172575398e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-4.3333333333333335e-02`, `x_rem=-5.1518900126128246e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-3.3898305084745762e-03`, `x_r_target=-4.0301616787062466e-03`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 2
- pauli term from int row: `(2.1322111528987456e-02)*X(q(0))*X(q(1))*Z(q(3))`
- int observable row: `[1, 1, 0, 3, 0, 0]`
- Hamiltonian weight w_2: `2.1322111528987456e-02`
- OGM effective shots used for this term: `606`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-4.7957371225577264e-02`, `x_rem=-5.7016408118819341e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0819672131147541e-01`, `x_rem=1.2863483259739461e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-5.6939501779359428e-02`, `x_rem=-6.7695242432361233e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-7.2463768115942030e-03`, `x_r_target=-8.6152006899880007e-03`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 3
- pauli term from int row: `(1.1735799722042688e-02)*Z(q(3))`
- int observable row: `[0, 0, 0, 3, 0, 0]`
- Hamiltonian weight w_3: `1.1735799722042688e-02`
- OGM effective shots used for this term: `4893`
- fitted unmit coeffs: `a_u=-1.9700041633529272e-01`, `b_u=-1.1277293826088359e+00`
- fitted rem coeffs: `a_r=-1.8344678769144515e-01`, `b_r=-1.1277293826088508e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-6.5443738470998158e-01`, `x_rem=-7.0278928770401805e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-6.6094853683148336e-01`, `x_rem=-7.0978150432934217e-01`, `y_exact=-9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=-6.5459036640592838e-01`, `x_rem=-7.0295357217131493e-01`, `y_exact=-9.9755099946510994e-01`
- target x values: `x_u_target=-6.7087062652563056e-01`, `x_r_target=-7.2043666937889883e-01`
- target contribution to E_cdr_unmit: `-1.1683781844543441e-02`
- target contribution to E_cdr_rem: `-1.1683781844543443e-02`

### term 4
- pauli term from int row: `(2.1322111528987456e-02)*Z(q(1))*Y(q(3))*Y(q(4))`
- int observable row: `[0, 3, 0, 2, 2, 0]`
- Hamiltonian weight w_4: `2.1322111528987456e-02`
- OGM effective shots used for this term: `542`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=3.9861351819757362e-02`, `x_rem=4.9158544301663470e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.1358885017421602e-02`, `x_rem=3.8672976906307888e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-3.4129692832764505e-03`, `x_rem=-4.2090043125181508e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=0.0000000000000000e+00`, `x_r_target=0.0000000000000000e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 5
- pauli term from int row: `(2.1322111528987456e-02)*Z(q(1))*X(q(3))*X(q(4))`
- int observable row: `[0, 3, 0, 1, 1, 0]`
- Hamiltonian weight w_5: `2.1322111528987456e-02`
- OGM effective shots used for this term: `592`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=3.6912751677852351e-02`, `x_rem=4.5522217782704703e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.8823529411764705e-02`, `x_rem=7.2543427268695199e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-4.7120418848167540e-02`, `x_rem=-5.8110703518902428e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.9962546816479401e-02`, `x_r_target=3.6950959208024534e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 6
- pauli term from int row: `(-1.4549167512868025e-01)*Z(q(1))`
- int observable row: `[0, 3, 0, 0, 0, 0]`
- Hamiltonian weight w_6: `-1.4549167512868025e-01`
- OGM effective shots used for this term: `5084`
- fitted unmit coeffs: `a_u=-1.3806877066276738e-08`, `b_u=1.0000000092942767e+00`
- fitted rem coeffs: `a_r=-1.3110997151967316e-08`, `b_r=1.0000000092942676e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=6.7321016166281755e-01`, `x_rem=7.0894077681425638e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=6.8742701440249121e-01`, `x_rem=7.2391218871365992e-01`, `y_exact=9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=6.8861105684704049e-01`, `x_rem=7.2515907418601577e-01`, `y_exact=9.9999999979455567e-01`
- target x values: `x_u_target=6.7889730149485539e-01`, `x_r_target=7.1492976147309995e-01`
- target contribution to E_cdr_unmit: `-1.4549167511716096e-01`
- target contribution to E_cdr_rem: `-1.4549167511716102e-01`

### term 7
- pauli term from int row: `(-1.4549167512868022e-01)*Z(q(4))`
- int observable row: `[0, 0, 0, 0, 3, 0]`
- Hamiltonian weight w_7: `-1.4549167512868022e-01`
- OGM effective shots used for this term: `5114`
- fitted unmit coeffs: `a_u=3.3716345186091133e-09`, `b_u=9.9999999756286961e-01`
- fitted rem coeffs: `a_r=3.0917840644201779e-09`, `b_r=9.9999999756287250e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=6.9757826343768459e-01`, `x_rem=7.6071784453400726e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=6.9158878504672894e-01`, `x_rem=7.5418624323525540e-01`, `y_exact=9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=6.5746721471912306e-01`, `x_rem=7.1697624287799711e-01`, `y_exact=9.9999999979455567e-01`
- target x values: `x_u_target=6.8707482993197277e-01`, `x_r_target=7.4926371857358010e-01`
- target contribution to E_cdr_unmit: `-1.4549167511113897e-01`
- target contribution to E_cdr_rem: `-1.4549167511113889e-01`

### term 8
- pauli term from int row: `(-1.7148070549056224e-01)*Z(q(2))`
- int observable row: `[0, 0, 3, 0, 0, 0]`
- Hamiltonian weight w_8: `-1.7148070549056224e-01`
- OGM effective shots used for this term: `6383`
- fitted unmit coeffs: `a_u=9.9680864609598643e-02`, `b_u=9.1787767856938440e-01`
- fitted rem coeffs: `a_r=9.4457587304063992e-02`, `b_r=9.1787767856937785e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=8.2345562872373790e-01`, `x_rem=8.6899074369326501e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=7.9730153749607779e-01`, `x_rem=8.4139039414951222e-01`, `y_exact=9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=8.0166326690726497e-01`, `x_rem=8.4599331670247480e-01`, `y_exact=9.9755099946510994e-01`
- target x values: `x_u_target=8.0864390854995305e-01`, `x_r_target=8.5335997103203143e-01`
- target contribution to E_cdr_unmit: `-1.7122074117572997e-01`
- target contribution to E_cdr_rem: `-1.7122074117573005e-01`

### term 9
- pauli term from int row: `(-1.7148070549056224e-01)*Z(q(5))`
- int observable row: `[0, 0, 0, 0, 0, 3]`
- Hamiltonian weight w_9: `-1.7148070549056224e-01`
- OGM effective shots used for this term: `6383`
- fitted unmit coeffs: `a_u=-6.9185130230153838e-02`, `b_u=1.0498989663728220e+00`
- fitted rem coeffs: `a_r=-6.2460335571780906e-02`, `b_r=1.0498989663728204e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=7.4443399184697401e-01`, `x_rem=8.2458350891335197e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=7.4239096328835896e-01`, `x_rem=8.2232051759898006e-01`, `y_exact=9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=7.4768554840734347e-01`, `x_rem=8.2818514444765545e-01`, `y_exact=9.9755099946510994e-01`
- target x values: `x_u_target=7.6103977450673344e-01`, `x_r_target=8.4297715386213301e-01`
- target contribution to E_cdr_unmit: `-1.7100850429571193e-01`
- target contribution to E_cdr_rem: `-1.7100850429571193e-01`

### term 10
- pauli term from int row: `(1.1183806637694067e-01)*Z(q(0))*Z(q(3))`
- int observable row: `[3, 0, 0, 3, 0, 0]`
- Hamiltonian weight w_10: `1.1183806637694067e-01`
- OGM effective shots used for this term: `3429`
- fitted unmit coeffs: `a_u=-1.4296723827814886e-08`, `b_u=1.0000000074566853e+00`
- fitted rem coeffs: `a_r=-1.2663417147313023e-08`, `b_r=1.0000000074566779e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=5.2375434530706833e-01`, `x_rem=5.9130674116275961e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.3135888501742157e-01`, `x_rem=5.9989209350295924e-01`, `y_exact=9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=5.3832494820952947e-01`, `x_rem=6.0775662037851974e-01`, `y_exact=9.9999999979455567e-01`
- target x values: `x_u_target=5.4024319629415174e-01`, `x_r_target=6.0992227882856032e-01`
- target contribution to E_cdr_unmit: `1.1183806634707739e-01`
- target contribution to E_cdr_rem: `1.1183806634707742e-01`

### term 11
- pauli term from int row: `(1.5264219777698216e-02)*Y(q(0))*Y(q(1))`
- int observable row: `[2, 2, 0, 0, 0, 0]`
- Hamiltonian weight w_11: `1.5264219777698216e-02`
- OGM effective shots used for this term: `1382`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.6141429669485011e-02`, `x_rem=-1.7870201544607032e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-3.9640987284966345e-02`, `x_rem=-4.3886597824029941e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=3.7089871611982884e-02`, `x_rem=4.1062253749601124e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.3758146270818247e-02`, `x_r_target=-1.5231664838493184e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 12
- pauli term from int row: `(1.5264219777698216e-02)*X(q(0))*X(q(1))`
- int observable row: `[1, 1, 0, 0, 0, 0]`
- Hamiltonian weight w_12: `1.5264219777698216e-02`
- OGM effective shots used for this term: `1432`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=2.8653295128939830e-02`, `x_rem=3.1722106985310097e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.6100981767180924e-03`, `x_rem=6.2109482961420692e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=9.5098756400877841e-03`, `x_rem=1.0528397907267889e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=4.1237113402061855e-02`, `x_r_target=4.5653671496384426e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 13
- pauli term from int row: `(1.5264219777698216e-02)*Z(q(0))*Z(q(1))*Y(q(3))*Y(q(4))`
- int observable row: `[3, 3, 0, 2, 2, 0]`
- Hamiltonian weight w_13: `1.5264219777698216e-02`
- OGM effective shots used for this term: `542`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.9064124783362217e-02`, `x_rem=-2.4716787367823619e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.7421602787456445e-02`, `x_rem=2.2587244712122614e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-2.3890784982935155e-02`, `x_rem=-3.0974590188842566e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.6778523489932886e-02`, `x_r_target=2.1753487356977158e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 14
- pauli term from int row: `(1.5264219777698216e-02)*Z(q(0))*Z(q(1))*X(q(3))*X(q(4))`
- int observable row: `[3, 3, 0, 1, 1, 0]`
- Hamiltonian weight w_14: `1.5264219777698216e-02`
- OGM effective shots used for this term: `592`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-3.0201342281879196e-02`, `x_rem=-3.9156277242558900e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-3.6764705882352942e-02`, `x_rem=-4.7665729649847027e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.2216404886561954e-02`, `x_rem=1.5838664791153364e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-3.7453183520599252e-02`, `x_r_target=-4.8558346309956507e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 15
- pauli term from int row: `(5.7263977866495088e-03)*Y(q(0))*X(q(1))*X(q(3))*Y(q(4))`
- int observable row: `[2, 1, 0, 1, 2, 0]`
- Hamiltonian weight w_15: `5.7263977866495088e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.2965078464758386e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.2965078464758386e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 16
- pauli term from int row: `(-5.7263977866495088e-03)*Y(q(0))*X(q(1))*Y(q(3))*X(q(4))`
- int observable row: `[2, 1, 0, 2, 1, 0]`
- Hamiltonian weight w_16: `-5.7263977866495088e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.2965078464758386e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.2965078464758386e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 17
- pauli term from int row: `(-5.7263977866495088e-03)*X(q(0))*Y(q(1))*X(q(3))*Y(q(4))`
- int observable row: `[1, 2, 0, 1, 2, 0]`
- Hamiltonian weight w_17: `-5.7263977866495088e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.2965078464758386e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.2965078464758386e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 18
- pauli term from int row: `(5.7263977866495088e-03)*X(q(0))*Y(q(1))*Y(q(3))*X(q(4))`
- int observable row: `[1, 2, 0, 2, 1, 0]`
- Hamiltonian weight w_18: `5.7263977866495088e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.2965078464758391e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758391e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758391e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758391e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.2965078464758391e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 19
- pauli term from int row: `(5.2719059284356143e-03)*Y(q(0))*X(q(2))*X(q(3))*Y(q(5))`
- int observable row: `[2, 0, 1, 1, 0, 2]`
- Hamiltonian weight w_19: `5.2719059284356143e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=9.5337142957241028e-01`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.2730512888687209e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963106e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963106e+00`, `y_exact=-6.9942855641384583e-02`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963106e+00`, `y_exact=-6.9942855641384583e-02`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3196798592963106e+00`
- target contribution to E_cdr_unmit: `-2.4582143687168782e-04`
- target contribution to E_cdr_rem: `-2.4582143687168782e-04`

### term 20
- pauli term from int row: `(-5.2719059284356143e-03)*Y(q(0))*X(q(2))*Y(q(3))*X(q(5))`
- int observable row: `[2, 0, 1, 2, 0, 1]`
- Hamiltonian weight w_20: `-5.2719059284356143e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0466285704275897e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.3663084297239003e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963106e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963106e+00`, `y_exact=6.9942855641384583e-02`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963106e+00`, `y_exact=6.9942855641384583e-02`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3196798592963106e+00`
- target contribution to E_cdr_unmit: `-2.4582143687168782e-04`
- target contribution to E_cdr_rem: `-2.4582143687168782e-04`

### term 21
- pauli term from int row: `(-5.2719059284356143e-03)*X(q(0))*Y(q(2))*X(q(3))*Y(q(5))`
- int observable row: `[1, 0, 2, 1, 0, 2]`
- Hamiltonian weight w_21: `-5.2719059284356143e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0466285704275897e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.3663084297239001e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963104e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963104e+00`, `y_exact=6.9942855641384583e-02`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963104e+00`, `y_exact=6.9942855641384583e-02`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3196798592963104e+00`
- target contribution to E_cdr_unmit: `-2.4582143687168782e-04`
- target contribution to E_cdr_rem: `-2.4582143687168782e-04`

### term 22
- pauli term from int row: `(5.2719059284356143e-03)*X(q(0))*Y(q(2))*Y(q(3))*X(q(5))`
- int observable row: `[1, 0, 2, 2, 0, 1]`
- Hamiltonian weight w_22: `5.2719059284356143e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=9.5337142957241028e-01`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.2730512888687213e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963110e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963110e+00`, `y_exact=-6.9942855641384583e-02`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963110e+00`, `y_exact=-6.9942855641384583e-02`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3196798592963110e+00`
- target contribution to E_cdr_unmit: `-2.4582143687168782e-04`
- target contribution to E_cdr_rem: `-2.4582143687168782e-04`

### term 23
- pauli term from int row: `(4.7435147736005108e-02)*Z(q(0))*Z(q(1))`
- int observable row: `[3, 3, 0, 0, 0, 0]`
- Hamiltonian weight w_23: `4.7435147736005108e-02`
- OGM effective shots used for this term: `4563`
- fitted unmit coeffs: `a_u=1.0532530239307883e-01`, `b_u=-9.3037379592392655e-01`
- fitted rem coeffs: `a_r=9.5136082083437948e-02`, `b_r=-9.3037379592391878e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-6.5708108108108110e-01`, `x_rem=-7.2745547268749644e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-6.3393248575186323e-01`, `x_rem=-7.0182762729349746e-01`, `y_exact=-9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=-6.4565888056412513e-01`, `x_rem=-7.1480993697594819e-01`, `y_exact=-9.9755099946510994e-01`
- target x values: `x_u_target=-6.0558464223385688e-01`, `x_r_target=-6.7044368625514972e-01`
- target contribution to E_cdr_unmit: `-4.7157992776874547e-02`
- target contribution to E_cdr_rem: `-4.7157992776874519e-02`

### term 24
- pauli term from int row: `(5.3161545522654612e-02)*Z(q(0))*Z(q(4))`
- int observable row: `[3, 0, 0, 0, 3, 0]`
- Hamiltonian weight w_24: `5.3161545522654612e-02`
- OGM effective shots used for this term: `3723`
- fitted unmit coeffs: `a_u=8.4651332756091324e-02`, `b_u=-9.5563617283269964e-01`
- fitted rem coeffs: `a_r=7.3837158857033167e-02`, `b_r=-9.5563617283269964e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-5.2119434817382038e-01`, `x_rem=-5.9752835673542881e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-5.0254350736278453e-01`, `x_rem=-5.7614591791850656e-01`, `y_exact=-9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=-4.9063263643768668e-01`, `x_rem=-5.6249058348117331e-01`, `y_exact=-9.9755099946510994e-01`
- target x values: `x_u_target=-4.9041533546325877e-01`, `x_r_target=-5.6224145665425773e-01`
- target contribution to E_cdr_unmit: `-5.3010060879132952e-02`
- target contribution to E_cdr_rem: `-5.3010060879132931e-02`

### term 25
- pauli term from int row: `(-3.8063028881712008e-03)*Y(q(0))*Y(q(1))*Z(q(3))*Z(q(4))`
- int observable row: `[2, 2, 0, 3, 3, 0]`
- Hamiltonian weight w_25: `-3.8063028881712008e-03`
- OGM effective shots used for this term: `561`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-7.7504725897920609e-02`, `x_rem=-1.0048548526561323e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.2544802867383512e-01`, `x_rem=1.6264435350055326e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-2.9999999999999999e-02`, `x_rem=-3.8895235394275168e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=4.7457627118644069e-02`, `x_r_target=6.1529185934446590e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 26
- pauli term from int row: `(-3.8063028881712008e-03)*X(q(0))*X(q(1))*Z(q(3))*Z(q(4))`
- int observable row: `[1, 1, 0, 3, 3, 0]`
- Hamiltonian weight w_26: `-3.8063028881712008e-03`
- OGM effective shots used for this term: `606`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-8.3481349911190050e-02`, `x_rem=-1.0823422519425296e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=8.5245901639344257e-02`, `x_rem=1.1052198035531739e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-5.3380782918149468e-02`, `x_rem=-6.9208603904404212e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.4492753623188406e-02`, `x_r_target=-1.8789968789504913e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 27
- pauli term from int row: `(5.1256723004769753e-03)*Y(q(0))*Z(q(1))*X(q(2))*Z(q(3))*X(q(4))*Y(q(5))`
- int observable row: `[2, 3, 1, 3, 1, 2]`
- Hamiltonian weight w_27: `5.1256723004769753e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.5155090949116972e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116972e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116972e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116972e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.5155090949116972e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 28
- pauli term from int row: `(-5.1256723004769753e-03)*Y(q(0))*Z(q(1))*X(q(2))*Z(q(3))*Y(q(4))*X(q(5))`
- int observable row: `[2, 3, 1, 3, 2, 1]`
- Hamiltonian weight w_28: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116974e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116974e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116974e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.5155090949116974e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 29
- pauli term from int row: `(-5.1256723004769753e-03)*X(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*X(q(4))*Y(q(5))`
- int observable row: `[1, 3, 2, 3, 1, 2]`
- Hamiltonian weight w_29: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.5155090949116965e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 30
- pauli term from int row: `(5.1256723004769753e-03)*X(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*Y(q(4))*X(q(5))`
- int observable row: `[1, 3, 2, 3, 2, 1]`
- Hamiltonian weight w_30: `5.1256723004769753e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.5155090949116972e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116972e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116972e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116972e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.5155090949116972e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 31
- pauli term from int row: `(5.5376587099558762e-02)*Z(q(0))*Z(q(2))`
- int observable row: `[3, 0, 3, 0, 0, 0]`
- Hamiltonian weight w_31: `5.5376587099558762e-02`
- OGM effective shots used for this term: `4563`
- fitted unmit coeffs: `a_u=3.3768843238712634e-08`, `b_u=-9.9999997617738479e-01`
- fitted rem coeffs: `a_r=3.0437739956688122e-08`, `b_r=-9.9999997617742109e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-7.0464864864864862e-01`, `x_rem=-7.8176411215196118e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-6.9837790442788250e-01`, `x_rem=-7.7480710911551087e-01`, `y_exact=-9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=-7.0118995152049357e-01`, `x_rem=-7.7792690151545441e-01`, `y_exact=-9.9999999979455567e-01`
- target x values: `x_u_target=-7.1640488656195467e-01`, `x_r_target=-7.9480693131037194e-01`
- target contribution to E_cdr_unmit: `-5.5376587120023128e-02`
- target contribution to E_cdr_rem: `-5.5376587120023052e-02`

### term 32
- pauli term from int row: `(1.4370460325369306e-03)*Y(q(0))*Y(q(1))*Z(q(2))*Z(q(3))`
- int observable row: `[2, 2, 3, 3, 0, 0]`
- Hamiltonian weight w_32: `1.4370460325369306e-03`
- OGM effective shots used for this term: `561`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0018903591682420e-01`, `x_rem=-1.2570125990677525e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0394265232974910e-01`, `x_rem=1.3041070049569470e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-6.6666666666666666e-02`, `x_rem=-8.3642725145514588e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=0.0000000000000000e+00`, `x_r_target=-2.4086194432545768e-17`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 33
- pauli term from int row: `(1.4370460325369306e-03)*X(q(0))*X(q(1))*Z(q(2))*Z(q(3))`
- int observable row: `[1, 1, 3, 3, 0, 0]`
- Hamiltonian weight w_33: `1.4370460325369306e-03`
- OGM effective shots used for this term: `606`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-5.5062166962699825e-02`, `x_rem=-6.9083245457663062e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=7.5409836065573776e-02`, `x_rem=9.4612262869516434e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-3.9145907473309607e-02`, `x_rem=-4.9114055690426754e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.4492753623188406e-02`, `x_r_target=-1.8183201118590158e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 34
- pauli term from int row: `(6.0648493027994375e-02)*Z(q(0))*Z(q(5))`
- int observable row: `[3, 0, 0, 0, 0, 3]`
- Hamiltonian weight w_34: `6.0648493027994375e-02`
- OGM effective shots used for this term: `4563`
- fitted unmit coeffs: `a_u=-2.7688688752924497e-10`, `b_u=-1.0000000000261906e+00`
- fitted rem coeffs: `a_r=-2.3777387915992950e-10`, `b_r=-1.0000000000261902e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-5.8875675675675676e-01`, `x_rem=-6.8560268897654919e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-5.7606313020604993e-01`, `x_rem=-6.7082105904847988e-01`, `y_exact=-9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=-6.0290877038342883e-01`, `x_rem=-7.0208259937337836e-01`, `y_exact=-9.9999999979455567e-01`
- target x values: `x_u_target=-5.7547993019197208e-01`, `x_r_target=-6.7014192714336929e-01`
- target contribution to E_cdr_unmit: `-6.0648493019918891e-02`
- target contribution to E_cdr_rem: `-6.0648493019918905e-02`

### term 35
- pauli term from int row: `(-3.6886262679400451e-03)*Y(q(0))*Y(q(1))*Z(q(3))*Z(q(5))`
- int observable row: `[2, 2, 0, 3, 0, 3]`
- Hamiltonian weight w_35: `-3.6886262679400451e-03`
- OGM effective shots used for this term: `561`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-5.1039697542533083e-02`, `x_rem=-6.7214198906689129e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.2258064516129031e-02`, `x_rem=4.2480658568265932e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-6.6666666666666666e-02`, `x_rem=-8.7793361041082818e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=3.7288135593220341e-02`, `x_r_target=4.9104761260266701e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 36
- pauli term from int row: `(-3.6886262679400451e-03)*X(q(0))*X(q(1))*Z(q(3))*Z(q(5))`
- int observable row: `[1, 1, 0, 3, 0, 3]`
- Hamiltonian weight w_36: `-3.6886262679400451e-03`
- OGM effective shots used for this term: `606`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-5.1509769094138541e-02`, `x_rem=-6.7833236328367727e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=8.5245901639344257e-02`, `x_rem=1.1226036329843386e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0320284697508897e-01`, `x_rem=-1.3590787207427418e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.8115942028985508e-02`, `x_r_target=-2.3856891587250774e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 37
- pauli term from int row: `(5.3161545522654612e-02)*Z(q(1))*Z(q(3))`
- int observable row: `[0, 3, 0, 3, 0, 0]`
- Hamiltonian weight w_37: `5.3161545522654612e-02`
- OGM effective shots used for this term: `3726`
- fitted unmit coeffs: `a_u=-1.6435449398238872e-01`, `b_u=-1.0793706521751578e+00`
- fitted rem coeffs: `a_r=-1.4533334079467758e-01`, `b_r=-1.0793706521751667e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-4.8296804858727227e-01`, `x_rem=-5.4617865935782928e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-4.9722735674676527e-01`, `x_rem=-5.6230421846407463e-01`, `y_exact=-9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=-4.9837662337662336e-01`, `x_rem=-5.6360390052166975e-01`, `y_exact=-9.9755099946510994e-01`
- target x values: `x_u_target=-5.0927397986221512e-01`, `x_r_target=-5.7592749744128957e-01`
- target contribution to E_cdr_unmit: `-5.2931312699443261e-02`
- target contribution to E_cdr_rem: `-5.2931312699443248e-02`

### term 38
- pauli term from int row: `(-3.8063028881712008e-03)*Y(q(3))*Y(q(4))`
- int observable row: `[0, 0, 0, 2, 2, 0]`
- Hamiltonian weight w_38: `-3.8063028881712008e-03`
- OGM effective shots used for this term: `1354`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=2.8446389496717725e-02`, `x_rem=3.3313084717925615e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=0.0000000000000000e+00`, `x_rem=-6.3805920955468766e-19`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-2.1505376344086021e-03`, `x_rem=-2.5184581829763432e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=9.1484869809992965e-03`, `x_r_target=1.0713638083104863e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 39
- pauli term from int row: `(-3.8063028881712008e-03)*X(q(3))*X(q(4))`
- int observable row: `[0, 0, 0, 1, 1, 0]`
- Hamiltonian weight w_39: `-3.8063028881712008e-03`
- OGM effective shots used for this term: `1427`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=9.2395167022032692e-03`, `x_rem=1.0820241447115847e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-2.2710068130204391e-03`, `x_rem=-2.6595375967085558e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-3.7199124726477024e-02`, `x_rem=-4.3563264631133483e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.2970903522205207e-02`, `x_r_target=2.6900835874823885e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 40
- pauli term from int row: `(-5.1256723004769753e-03)*X(q(1))*X(q(2))*Y(q(3))*Y(q(5))`
- int observable row: `[0, 1, 1, 2, 0, 2]`
- Hamiltonian weight w_40: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `320`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=2.5495750708215296e-02`, `x_rem=3.3702919909811023e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-8.7499999999999994e-02`, `x_rem=-1.1566654874603199e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=3.9755351681957186e-02`, `x_rem=5.2552735122705634e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-7.4433656957928807e-02`, `x_r_target=-9.8394105267082213e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 41
- pauli term from int row: `(-5.1256723004769753e-03)*Y(q(1))*Y(q(2))*Y(q(3))*Y(q(5))`
- int observable row: `[0, 2, 2, 2, 0, 2]`
- Hamiltonian weight w_41: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `314`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=0.0000000000000000e+00`, `x_rem=6.1679056923619804e-18`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=2.5477707006369428e-02`, `x_rem=3.3679067878735418e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0309278350515464e-02`, `x_rem=-1.3627870249900682e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=9.9667774086378731e-03`, `x_r_target=1.3175117085452130e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 42
- pauli term from int row: `(-5.1256723004769753e-03)*X(q(1))*X(q(2))*X(q(3))*X(q(5))`
- int observable row: `[0, 1, 1, 1, 0, 1]`
- Hamiltonian weight w_42: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `316`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=4.0000000000000001e-02`, `x_rem=5.2876136569614603e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.3826366559485531e-01`, `x_rem=1.8277121161522739e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-5.7692307692307696e-02`, `x_rem=-7.6263658513867252e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-4.7021943573667714e-02`, `x_r_target=-6.2158467754249117e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 43
- pauli term from int row: `(-5.1256723004769753e-03)*Y(q(1))*Y(q(2))*X(q(3))*X(q(5))`
- int observable row: `[0, 2, 2, 1, 0, 1]`
- Hamiltonian weight w_43: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `338`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=7.2847682119205295e-02`, `x_rem=9.6297599712874285e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0437710437710437e-01`, `x_rem=-1.3797645064461728e-01`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.2987012987012988e-02`, `x_rem=-1.7167576808316438e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=3.8709677419354840e-02`, `x_r_target=5.1170454744788353e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 44
- pauli term from int row: `(4.7435147736005108e-02)*Z(q(3))*Z(q(4))`
- int observable row: `[0, 0, 0, 3, 3, 0]`
- Hamiltonian weight w_44: `4.7435147736005108e-02`
- OGM effective shots used for this term: `4596`
- fitted unmit coeffs: `a_u=-1.4998559286289933e-02`, `b_u=-1.0075674410439972e+00`
- fitted rem coeffs: `a_r=-1.2807425759579542e-02`, `b_r=-1.0075674410439972e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-6.1003521126760563e-01`, `x_rem=-7.1440189892008099e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-6.3139635732870769e-01`, `x_rem=-7.3941757510941197e-01`, `y_exact=-9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=-5.9876679145562650e-01`, `x_rem=-7.0120564342069946e-01`, `y_exact=-9.9755099946510994e-01`
- target x values: `x_u_target=-6.2619669277632728e-01`, `x_r_target=-7.3332833605999825e-01`
- target contribution to E_cdr_unmit: `-4.7348597224980825e-02`
- target contribution to E_cdr_rem: `-4.7348597224980825e-02`

### term 45
- pauli term from int row: `(6.0648493027994375e-02)*Z(q(2))*Z(q(3))`
- int observable row: `[0, 0, 3, 3, 0, 0]`
- Hamiltonian weight w_45: `6.0648493027994375e-02`
- OGM effective shots used for this term: `4596`
- fitted unmit coeffs: `a_u=1.5333896273218640e-08`, `b_u=-9.9999999100675752e-01`
- fitted rem coeffs: `a_r=1.3530706136237184e-08`, `b_r=-9.9999999100675896e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-5.7834507042253525e-01`, `x_rem=-6.5541898762162121e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-5.7935819601040761e-01`, `x_rem=-6.5656712872473788e-01`, `y_exact=-9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=-5.7498348381413789e-01`, `x_rem=-6.5160941474833900e-01`, `y_exact=-9.9999999979455567e-01`
- target x values: `x_u_target=-5.8006962576153176e-01`, `x_r_target=-6.5737336809824032e-01`
- target contribution to E_cdr_unmit: `-6.0648493022019592e-02`
- target contribution to E_cdr_rem: `-6.0648493022019578e-02`

### term 46
- pauli term from int row: `(-3.6886262679400451e-03)*Z(q(1))*Z(q(2))*Y(q(3))*Y(q(4))`
- int observable row: `[0, 3, 3, 2, 2, 0]`
- Hamiltonian weight w_46: `-3.6886262679400451e-03`
- OGM effective shots used for this term: `542`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=3.9861351819757362e-02`, `x_rem=5.1876893522228226e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.4843205574912892e-03`, `x_rem=4.5346110532231022e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=2.0477815699658702e-02`, `x_rem=2.6650512742833362e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-5.0335570469798654e-02`, `x_r_target=-6.5508391221897472e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 47
- pauli term from int row: `(-3.6886262679400451e-03)*Z(q(1))*Z(q(2))*X(q(3))*X(q(4))`
- int observable row: `[0, 3, 3, 1, 1, 0]`
- Hamiltonian weight w_47: `-3.6886262679400451e-03`
- OGM effective shots used for this term: `592`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=2.6845637583892617e-02`, `x_rem=3.4937808651678649e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=4.0441176470588237e-02`, `x_rem=5.2631496672887239e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-3.3158813263525308e-02`, `x_rem=-4.3153986166187736e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=4.1198501872659173e-02`, `x_r_target=5.3617105224813957e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 48
- pauli term from int row: `(5.5376587099558762e-02)*Z(q(3))*Z(q(5))`
- int observable row: `[0, 0, 0, 3, 0, 3]`
- Hamiltonian weight w_48: `5.5376587099558762e-02`
- OGM effective shots used for this term: `4596`
- fitted unmit coeffs: `a_u=-6.2663231929176359e-09`, `b_u=-1.0000000040190222e+00`
- fitted rem coeffs: `a_r=-5.2680185101407906e-09`, `b_r=-1.0000000040190220e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-6.5713028169014087e-01`, `x_rem=-7.8165833454441469e-01`, `y_exact=-1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-6.7476149176062450e-01`, `x_rem=-8.0263070894824029e-01`, `y_exact=-9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=-6.5778462893635758e-01`, `x_rem=-7.8243668245036735e-01`, `y_exact=-9.9999999979455567e-01`
- target x values: `x_u_target=-6.7362924281984338e-01`, `x_r_target=-8.0128389562065416e-01`
- target contribution to E_cdr_unmit: `-5.5376587088364036e-02`
- target contribution to E_cdr_rem: `-5.5376587088364029e-02`

### term 49
- pauli term from int row: `(1.4370460325369306e-03)*Z(q(1))*Y(q(3))*Y(q(4))*Z(q(5))`
- int observable row: `[0, 3, 0, 2, 2, 3]`
- Hamiltonian weight w_49: `1.4370460325369306e-03`
- OGM effective shots used for this term: `542`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=1.2131715771230503e-02`, `x_rem=1.6572104665275391e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.8327526132404179e-02`, `x_rem=5.2355972279991475e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-6.4846416382252553e-02`, `x_rem=-8.8581171840767456e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-6.7114093959731542e-03`, `x_r_target=-9.1678853229759400e-03`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 50
- pauli term from int row: `(1.4370460325369306e-03)*Z(q(1))*X(q(3))*X(q(4))*Z(q(5))`
- int observable row: `[0, 3, 0, 1, 1, 3]`
- Hamiltonian weight w_50: `1.4370460325369306e-03`
- OGM effective shots used for this term: `592`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=6.3758389261744972e-02`, `x_rem=8.7094910568271372e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.3088235294117647e-02`, `x_rem=4.5199022860701210e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.7452006980802793e-03`, `x_rem=2.3839701799710433e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.4981273408239701e-02`, `x_r_target=-2.0464642893234668e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 51
- pauli term from int row: `(8.1752954675092832e-02)*Z(q(1))*Z(q(4))`
- int observable row: `[0, 3, 0, 0, 3, 0]`
- Hamiltonian weight w_51: `8.1752954675092832e-02`
- OGM effective shots used for this term: `3653`
- fitted unmit coeffs: `a_u=-3.0657371203510159e-09`, `b_u=1.0000000014203532e+00`
- fitted rem coeffs: `a_r=-2.6695928194988669e-09`, `b_r=1.0000000014203534e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=5.0596529284164848e-01`, `x_rem=5.8104622693874752e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.1699755235246125e-01`, `x_rem=5.9371557966720234e-01`, `y_exact=9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=5.0096074663738677e-01`, `x_rem=5.7529904876137605e-01`, `y_exact=9.9999999979455567e-01`
- target x values: `x_u_target=5.0661983247770881e-01`, `x_r_target=5.8179789467425302e-01`
- target contribution to E_cdr_unmit: `8.1752954664235211e-02`
- target contribution to E_cdr_rem: `8.1752954664235211e-02`

### term 52
- pauli term from int row: `(1.0346900732713493e-02)*Y(q(1))*X(q(2))*X(q(4))*Y(q(5))`
- int observable row: `[0, 2, 1, 0, 1, 2]`
- Hamiltonian weight w_52: `1.0346900732713493e-02`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.3423734562057010e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057010e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057010e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057010e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.3423734562057010e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 53
- pauli term from int row: `(-1.0346900732713493e-02)*Y(q(1))*X(q(2))*Y(q(4))*X(q(5))`
- int observable row: `[0, 2, 1, 0, 2, 1]`
- Hamiltonian weight w_53: `-1.0346900732713493e-02`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.3423734562057017e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057017e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057017e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057017e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3423734562057017e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 54
- pauli term from int row: `(-1.0346900732713493e-02)*X(q(1))*Y(q(2))*X(q(4))*Y(q(5))`
- int observable row: `[0, 1, 2, 0, 1, 2]`
- Hamiltonian weight w_54: `-1.0346900732713493e-02`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.3423734562057013e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057013e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057013e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057013e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3423734562057013e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 55
- pauli term from int row: `(1.0346900732713493e-02)*X(q(1))*Y(q(2))*Y(q(4))*X(q(5))`
- int observable row: `[0, 1, 2, 0, 2, 1]`
- Hamiltonian weight w_55: `1.0346900732713493e-02`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.3423734562057013e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057013e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057013e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057013e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.3423734562057013e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 56
- pauli term from int row: `(5.9449198989561303e-02)*Z(q(1))*Z(q(2))`
- int observable row: `[0, 3, 3, 0, 0, 0]`
- Hamiltonian weight w_56: `5.9449198989561303e-02`
- OGM effective shots used for this term: `4563`
- fitted unmit coeffs: `a_u=1.6266681329929100e-01`, `b_u=8.9645806211230172e-01`
- fitted rem coeffs: `a_r=1.4637426143936200e-01`, `b_r=8.9645806211231083e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=6.3329729729729733e-01`, `x_rem=7.0378803082857833e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=6.1902674265672952e-01`, `x_rem=6.8792905654875913e-01`, `y_exact=9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=6.2714852357866901e-01`, `x_rem=6.9695485253157308e-01`, `y_exact=9.9755099946510994e-01`
- target x values: `x_u_target=6.0471204188481675e-01`, `x_r_target=6.7202102234245575e-01`
- target contribution to E_cdr_unmit: `5.9141528157229901e-02`
- target contribution to E_cdr_rem: `5.9141528157229929e-02`

### term 57
- pauli term from int row: `(6.9796099722274796e-02)*Z(q(1))*Z(q(5))`
- int observable row: `[0, 3, 0, 0, 0, 3]`
- Hamiltonian weight w_57: `6.9796099722274796e-02`
- OGM effective shots used for this term: `4563`
- fitted unmit coeffs: `a_u=2.2150066066275609e-01`, `b_u=8.7354091952414736e-01`
- fitted rem coeffs: `a_r=1.8989226830544634e-01`, `b_r=8.7354091952414392e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=5.6843243243243247e-01`, `x_rem=6.6305047830277453e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.5808855765015342e-01`, `x_rem=6.5098482066155716e-01`, `y_exact=9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=5.6412516527104450e-01`, `x_rem=6.5802624782508157e-01`, `y_exact=9.9755099946510994e-01`
- target x values: `x_u_target=5.4493891797556715e-01`, `x_r_target=6.3564636638224403e-01`
- target contribution to E_cdr_unmit: `6.9394440608783997e-02`
- target contribution to E_cdr_rem: `6.9394440608784011e-02`

### term 58
- pauli term from int row: `(6.9796099722274796e-02)*Z(q(2))*Z(q(4))`
- int observable row: `[0, 0, 3, 0, 3, 0]`
- Hamiltonian weight w_58: `6.9796099722274796e-02`
- OGM effective shots used for this term: `4596`
- fitted unmit coeffs: `a_u=7.7354888279120793e-02`, `b_u=9.5392700306813183e-01`
- fitted rem coeffs: `a_r=6.7217468286224227e-02`, `b_r=9.5392700306813616e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=5.8978873239436624e-01`, `x_rem=6.7873787373803474e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.7502168256721597e-01`, `x_rem=6.6174372744369425e-01`, `y_exact=9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=5.5868751376348824e-01`, `x_rem=6.4294611671601554e-01`, `y_exact=9.9755099946510994e-01`
- target x values: `x_u_target=5.8398607484769360e-01`, `x_r_target=6.7206008688159646e-01`
- target contribution to E_cdr_unmit: `6.9733365636908615e-02`
- target contribution to E_cdr_rem: `6.9733365636908587e-02`

### term 59
- pauli term from int row: `(5.9449198989561303e-02)*Z(q(4))*Z(q(5))`
- int observable row: `[0, 0, 0, 0, 3, 3]`
- Hamiltonian weight w_59: `5.9449198989561303e-02`
- OGM effective shots used for this term: `4596`
- fitted unmit coeffs: `a_u=-1.1891077542139620e-02`, `b_u=1.0052272841608088e+00`
- fitted rem coeffs: `a_r=-9.8442378262241521e-03`, `b_r=1.0052272841608083e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=5.7438380281690138e-01`, `x_rem=6.9381118770308381e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.9410234171725929e-01`, `x_rem=7.1762965686452684e-01`, `y_exact=9.9755099946510994e-01`
  - train[2] t_remaining=1: `x_unmit=5.6221096674741244e-01`, `x_rem=6.7910734367115300e-01`, `y_exact=9.9755099946510994e-01`
- target x values: `x_u_target=5.9660574412532641e-01`, `x_r_target=7.2065357325896862e-01`
- target contribution to E_cdr_unmit: `5.9338207275320881e-02`
- target contribution to E_cdr_rem: `5.9338207275320895e-02`

### term 60
- pauli term from int row: `(7.8236377789852360e-02)*Z(q(2))*Z(q(5))`
- int observable row: `[0, 0, 3, 0, 0, 3]`
- Hamiltonian weight w_60: `7.8236377789852360e-02`
- OGM effective shots used for this term: `6383`
- fitted unmit coeffs: `a_u=4.3399636153524038e-09`, `b_u=9.9999999711746423e-01`
- fitted rem coeffs: `a_r=3.7128099683680204e-09`, `b_r=9.9999999711746390e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=0: `x_unmit=6.4252116650987767e-01`, `x_rem=7.5105343493741739e-01`, `y_exact=1.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=6.1499843112645125e-01`, `x_rem=7.1888166219897287e-01`, `y_exact=9.9999999979455567e-01`
  - train[2] t_remaining=1: `x_unmit=6.4035775929703431e-01`, `x_rem=7.4852459308275876e-01`, `y_exact=9.9999999979455567e-01`
- target x values: `x_u_target=6.4547447541497027e-01`, `x_r_target=7.5450560571904257e-01`
- target contribution to E_cdr_unmit: `7.8236377783499567e-02`
- target contribution to E_cdr_rem: `7.8236377783499553e-02`

## Complete expanded energy expressions (all summing terms)
### E_cdr_unmit
`E_cdr_unmit = (-7.2476524580835413e+00) + (-1.1736481734387273e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-1.1683781844543441e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-1.4549167511716096e-01) + (-1.4549167511113897e-01) + (-1.7122074117572997e-01) + (-1.7100850429571193e-01) + (1.1183806634707739e-01) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-2.4582143687168782e-04) + (-2.4582143687168782e-04) + (-2.4582143687168782e-04) + (-2.4582143687168782e-04) + (-4.7157992776874547e-02) + (-5.3010060879132952e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-5.5376587120023128e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-6.0648493019918891e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-5.2931312699443261e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-4.7348597224980825e-02) + (-6.0648493022019592e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-5.5376587088364036e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (8.1752954664235211e-02) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (5.9141528157229901e-02) + (6.9394440608783997e-02) + (6.9733365636908615e-02) + (5.9338207275320881e-02) + (7.8236377783499567e-02)`

### E_cdr_rem
`E_cdr_rem = (-7.2476524580835413e+00) + (-1.1736481734387270e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-1.1683781844543443e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-1.4549167511716102e-01) + (-1.4549167511113889e-01) + (-1.7122074117573005e-01) + (-1.7100850429571193e-01) + (1.1183806634707742e-01) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-2.4582143687168782e-04) + (-2.4582143687168782e-04) + (-2.4582143687168782e-04) + (-2.4582143687168782e-04) + (-4.7157992776874519e-02) + (-5.3010060879132931e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-5.5376587120023052e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (-6.0648493019918905e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-5.2931312699443248e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-4.7348597224980825e-02) + (-6.0648493022019578e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-5.5376587088364029e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (8.1752954664235211e-02) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (5.9141528157229929e-02) + (6.9394440608784011e-02) + (6.9733365636908587e-02) + (5.9338207275320895e-02) + (7.8236377783499553e-02)`
