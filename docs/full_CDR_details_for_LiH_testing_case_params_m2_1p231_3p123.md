# full CDR details for LiH testing case (params = [-2, 1.231, 3.123])

## Configuration used
- bond_length: `2.2`
- target params [theta1, theta2, theta3]: `[-2.0, 1.231, 3.123]`
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
- training non-Clifford counts (t_remaining): `[1, 1, 1]`
- OGM file: `/Users/zacharyhe/shadowgrouping/haozhaowu/LiH/hamil_class/ogm_outputs/OGM_ogm_LiH_2.2.txt`

## Constant term (offset)
- Hamiltonian identity term: `(-7.2476524580835413) * I`
- offset: `-7.2476524580835413`
- E_cdr_unmit: `-6.7043819320045719`
- E_cdr_rem: `-6.7080710965827581`
- delta (unmit - rem): `3.6891645781862437e-03`
- exact noiseless energy (statevector): `-6.5105264960154816`
- exact noisy-trace energy (`Tr[H rho_noisy]`, gate-noise only): `-6.7323500329415982`

## Int encoding map
- `I=0, X=1, Y=2, Z=3`

## Per-term full details (Pauli term, training pairs, fits, shots, target contribution)
### term 0
- pauli term from int row: `(1.1735799722042688e-02)*Z(q(0))`
- int observable row: `[3, 0, 0, 0, 0, 0]`
- Hamiltonian weight w_0: `1.1735799722042688e-02`
- OGM effective shots used for this term: `4866`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.0871794871794871e-02`, `x_rem=-1.1429557266395074e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-7.2090628218331601e-03`, `x_rem=-7.5789138160568554e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-2.2460333814135588e-02`, `x_rem=-2.3612630166248639e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.1426780864842281e-01`, `x_r_target=-1.2013016047983906e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 1
- pauli term from int row: `(2.1322111528987456e-02)*Y(q(0))*Y(q(1))*Z(q(3))`
- int observable row: `[2, 2, 0, 3, 0, 0]`
- Hamiltonian weight w_1: `2.1322111528987456e-02`
- OGM effective shots used for this term: `576`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.5332197614991482e-02`, `x_rem=1.8228414407095564e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=7.6124567474048443e-02`, `x_rem=9.0504322819389513e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.2131715771230503e-02`, `x_rem=1.4423368919459971e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=8.2236842105263164e-02`, `x_r_target=9.7771192040982238e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 2
- pauli term from int row: `(2.1322111528987456e-02)*X(q(0))*X(q(1))*Z(q(3))`
- int observable row: `[1, 1, 0, 3, 0, 0]`
- Hamiltonian weight w_2: `2.1322111528987456e-02`
- OGM effective shots used for this term: `599`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.3084112149532711e-02`, `x_rem=-1.5555670778557776e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-4.3321299638989168e-02`, `x_rem=-5.1504593294657508e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-4.4217687074829932e-02`, `x_rem=-5.2570306251151264e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-2.2146507666098807e-02`, `x_r_target=-2.6329931921360253e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 3
- pauli term from int row: `(1.1735799722042688e-02)*Z(q(3))`
- int observable row: `[0, 0, 0, 3, 0, 0]`
- Hamiltonian weight w_3: `1.1735799722042688e-02`
- OGM effective shots used for this term: `4917`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-8.2884376295068382e-03`, `x_rem=-8.9008136055700616e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-3.1321779077051574e-03`, `x_rem=-3.3635931139445424e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.7149857084524295e-02`, `x_rem=1.8416942745408397e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-8.8669950738916259e-02`, `x_r_target=-9.5221167030623152e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 4
- pauli term from int row: `(2.1322111528987456e-02)*Z(q(1))*Y(q(3))*Y(q(4))`
- int observable row: `[0, 3, 0, 2, 2, 0]`
- Hamiltonian weight w_4: `2.1322111528987456e-02`
- OGM effective shots used for this term: `562`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=3.4602076124567475e-03`, `x_rem=4.2672604275703178e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-5.9999999999999998e-02`, `x_rem=-7.3994295814069078e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=5.4054054054054057e-03`, `x_rem=6.6661527760422676e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-8.1911262798634810e-02`, `x_r_target=-1.0101610350043562e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 5
- pauli term from int row: `(2.1322111528987456e-02)*Z(q(1))*X(q(3))*X(q(4))`
- int observable row: `[0, 3, 0, 1, 1, 0]`
- Hamiltonian weight w_5: `2.1322111528987456e-02`
- OGM effective shots used for this term: `578`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-5.2173913043478265e-03`, `x_rem=-6.4342865925277317e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-2.9605263157894735e-02`, `x_rem=-3.6510343329310413e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=3.1250000000000000e-02`, `x_rem=3.8538695736494329e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.9469026548672566e-02`, `x_r_target=2.4009948494240700e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 6
- pauli term from int row: `(-1.4549167512868025e-01)*Z(q(1))`
- int observable row: `[0, 3, 0, 0, 0, 0]`
- Hamiltonian weight w_6: `-1.4549167512868025e-01`
- OGM effective shots used for this term: `5126`
- fitted unmit coeffs: `a_u=1.5288155408746928e+00`, `b_u=-2.4621776659834714e-03`
- fitted rem coeffs: `a_r=1.4517632376146075e+00`, `b_r=-2.4621776659836535e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.5729453401494297e-03`, `x_rem=1.6564293809493457e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-2.7283372365339581e-01`, `x_rem=-2.8731436779001240e-01`, `y_exact=-4.1614673047138062e-01`
  - train[2] t_remaining=1: `x_unmit=-2.6831179005092048e-01`, `x_rem=-2.8255243265682450e-01`, `y_exact=-4.1614673047138062e-01`
- target x values: `x_u_target=-2.7114967462039047e-01`, `x_r_target=-2.8554093789004897e-01`
- target contribution to E_cdr_unmit: `6.0670030584275551e-02`
- target contribution to E_cdr_rem: `6.0670030584275565e-02`

### term 7
- pauli term from int row: `(-1.4549167512868022e-01)*Z(q(4))`
- int observable row: `[0, 0, 0, 0, 3, 0]`
- Hamiltonian weight w_7: `-1.4549167512868022e-01`
- OGM effective shots used for this term: `5145`
- fitted unmit coeffs: `a_u=1.0620161869604914e-02`, `b_u=-9.9294165500624987e-01`
- fitted rem coeffs: `a_r=9.7386884344275240e-03`, `b_r=-9.9294165500624987e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-6.4833431894342597e-01`, `x_rem=-7.0701670549991935e-01`, `y_exact=-9.9982691755849573e-01`
  - train[1] t_remaining=1: `x_unmit=-6.6494640730448595e-01`, `x_rem=-7.2513239618809833e-01`, `y_exact=-9.9999993586766500e-01`
  - train[2] t_remaining=1: `x_unmit=-6.6426203938389550e-01`, `x_rem=-7.2438608438810881e-01`, `y_exact=-9.9999993586766500e-01`
- target x values: `x_u_target=-6.5510445407962159e-01`, `x_r_target=-7.1439962276948932e-01`
- target contribution to E_cdr_unmit: `1.4547697615567490e-01`
- target contribution to E_cdr_rem: `1.4547697615567490e-01`

### term 8
- pauli term from int row: `(-1.7148070549056224e-01)*Z(q(2))`
- int observable row: `[0, 0, 3, 0, 0, 0]`
- Hamiltonian weight w_8: `-1.7148070549056224e-01`
- OGM effective shots used for this term: `6360`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-4.1022404544020195e-03`, `x_rem=-4.3290844812177588e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=2.0446681346335326e-02`, `x_rem=2.1577333628467067e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.7384494909945186e-02`, `x_rem=1.8345815650005547e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-2.8730582143417543e-01`, `x_r_target=-3.0319314207912135e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 9
- pauli term from int row: `(-1.7148070549056224e-01)*Z(q(5))`
- int observable row: `[0, 0, 0, 0, 0, 3]`
- Hamiltonian weight w_9: `-1.7148070549056224e-01`
- OGM effective shots used for this term: `6360`
- fitted unmit coeffs: `a_u=-2.6561563600758782e-15`, `b_u=2.4180623857972465e-17`
- fitted rem coeffs: `a_r=-2.3979779618765005e-15`, `b_r=2.4180623857972632e-17`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-8.5200378668349643e-03`, `x_rem=-9.4373481023869062e-03`, `y_exact=5.5511151231257827e-17`
  - train[1] t_remaining=1: `x_unmit=1.2582573136206355e-02`, `x_rem=1.3937276402532596e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=2.3492560689115116e-03`, `x_rem=2.6021888224541179e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0371881374548877e-01`, `x_r_target=-1.1488570419305351e-01`
- target contribution to E_cdr_unmit: `-5.1388310763032913e-17`
- target contribution to E_cdr_rem: `-5.1388310763032870e-17`

### term 10
- pauli term from int row: `(1.1183806637694067e-01)*Z(q(0))*Z(q(3))`
- int observable row: `[3, 0, 0, 3, 0, 0]`
- Hamiltonian weight w_10: `1.1183806637694067e-01`
- OGM effective shots used for this term: `3422`
- fitted unmit coeffs: `a_u=-8.1878822819698912e-06`, `b_u=1.0000041981643075e+00`
- fitted rem coeffs: `a_r=-7.2524776604345638e-06`, `b_r=1.0000041981643142e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=5.4164205552274070e-01`, `x_rem=6.1150155907551929e-01`, `y_exact=9.9999975588992185e-01`
  - train[1] t_remaining=1: `x_unmit=5.1796407185628746e-01`, `x_rem=5.8476965415756199e-01`, `y_exact=9.9999993586766500e-01`
  - train[2] t_remaining=1: `x_unmit=5.2405949256342954e-01`, `x_rem=5.9165124547351211e-01`, `y_exact=9.9999993586766500e-01`
- target x values: `x_u_target=5.1715905699791109e-01`, `x_r_target=5.8386081069543283e-01`
- target contribution to E_cdr_unmit: `1.1183806232021923e-01`
- target contribution to E_cdr_rem: `1.1183806232021924e-01`

### term 11
- pauli term from int row: `(1.5264219777698216e-02)*Y(q(0))*Y(q(1))`
- int observable row: `[2, 2, 0, 0, 0, 0]`
- Hamiltonian weight w_11: `1.5264219777698216e-02`
- OGM effective shots used for this term: `1347`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-2.3909985935302386e-02`, `x_rem=-2.6470782101806558e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=0.0000000000000000e+00`, `x_rem=1.8581138487450318e-17`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-3.2165832737669764e-02`, `x_rem=-3.5610842759420627e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-4.1924398625429550e-02`, `x_r_target=-4.6414566021324141e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 12
- pauli term from int row: `(1.5264219777698216e-02)*X(q(0))*X(q(1))`
- int observable row: `[1, 1, 0, 0, 0, 0]`
- Hamiltonian weight w_12: `1.5264219777698216e-02`
- OGM effective shots used for this term: `1415`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.4836795252225520e-02`, `x_rem=1.6425838780227336e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.2403918613413712e-02`, `x_rem=3.5874427997629889e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.4347202295552367e-02`, `x_rem=-1.5883809666962984e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.8240405503258507e-02`, `x_r_target=3.1264996247433438e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 13
- pauli term from int row: `(1.5264219777698216e-02)*Z(q(0))*Z(q(1))*Y(q(3))*Y(q(4))`
- int observable row: `[3, 3, 0, 2, 2, 0]`
- Hamiltonian weight w_13: `1.5264219777698216e-02`
- OGM effective shots used for this term: `562`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-2.7681660899653980e-02`, `x_rem=-3.5889490559884801e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e-02`, `x_rem=-1.2965078464758388e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.6216216216216217e-02`, `x_rem=2.1024451564473058e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.7064846416382253e-02`, `x_r_target=2.2124707277744685e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 14
- pauli term from int row: `(1.5264219777698216e-02)*Z(q(0))*Z(q(1))*X(q(3))*X(q(4))`
- int observable row: `[3, 3, 0, 1, 1, 0]`
- Hamiltonian weight w_14: `1.5264219777698216e-02`
- OGM effective shots used for this term: `578`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-6.4347826086956522e-02`, `x_rem=-8.3427461425401778e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-3.2894736842105261e-03`, `x_rem=-4.2648284423547241e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=3.1250000000000000e-02`, `x_rem=4.0515870202369957e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.7699115044247787e-03`, `x_r_target=2.2947041530545728e-03`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 15
- pauli term from int row: `(5.7263977866495088e-03)*Y(q(0))*X(q(1))*X(q(3))*Y(q(4))`
- int observable row: `[2, 1, 0, 1, 2, 0]`
- Hamiltonian weight w_15: `5.7263977866495088e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.2965078464758386e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.2965078464758386e+00`
- target contribution to E_cdr_unmit: `1.1452795573299018e-02`
- target contribution to E_cdr_rem: `1.4848639324865926e-02`

### term 16
- pauli term from int row: `(-5.7263977866495088e-03)*Y(q(0))*X(q(1))*Y(q(3))*X(q(4))`
- int observable row: `[2, 1, 0, 2, 1, 0]`
- Hamiltonian weight w_16: `-5.7263977866495088e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.2965078464758386e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
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
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.2965078464758386e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758386e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.2965078464758386e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 18
- pauli term from int row: `(5.7263977866495088e-03)*X(q(0))*Y(q(1))*Y(q(3))*X(q(4))`
- int observable row: `[1, 2, 0, 2, 1, 0]`
- Hamiltonian weight w_18: `5.7263977866495088e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0061971920646904e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.3027050385405294e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758391e+00`, `y_exact=-1.8591576194071480e-02`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758391e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.2965078464758391e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.2965078464758391e+00`
- target contribution to E_cdr_unmit: `-3.5487586922684906e-05`
- target contribution to E_cdr_rem: `-3.5487586922684906e-05`

### term 19
- pauli term from int row: `(5.2719059284356143e-03)*Y(q(0))*X(q(2))*X(q(3))*Y(q(5))`
- int observable row: `[2, 0, 1, 1, 0, 2]`
- Hamiltonian weight w_19: `5.2719059284356143e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.2412491604922118e-07`
- fitted rem coeffs: `a_r=3.7887972178658197e-01`, `b_r=-4.9999993793754216e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963102e+00`, `y_exact=-9.9999975588992185e-01`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963104e+00`, `y_exact=-9.9999993586766500e-01`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963104e+00`, `y_exact=-9.9999993586766500e-01`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3196798592963104e+00`
- target contribution to E_cdr_unmit: `-5.2719052740607336e-03`
- target contribution to E_cdr_rem: `-5.2719052740607328e-03`

### term 20
- pauli term from int row: `(-5.2719059284356143e-03)*Y(q(0))*X(q(2))*Y(q(3))*X(q(5))`
- int observable row: `[2, 0, 1, 2, 0, 1]`
- Hamiltonian weight w_20: `-5.2719059284356143e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=-8.6509154584702130e-05`, `b_u=-9.9991342671308003e-01`
- fitted rem coeffs: `a_r=-6.5553136978924135e-05`, `b_r=-9.9991342671308037e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963108e+00`, `y_exact=-9.9982691755849573e-01`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3196798592963106e+00`, `y_exact=-9.9999993586766500e-01`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3196798592963106e+00`, `y_exact=-9.9999993586766500e-01`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.3196798592963106e+00`
- target contribution to E_cdr_unmit: `5.2719055903359758e-03`
- target contribution to E_cdr_rem: `5.2719055903359775e-03`

### term 21
- pauli term from int row: `(-5.2719059284356143e-03)*X(q(0))*Y(q(2))*X(q(3))*Y(q(5))`
- int observable row: `[1, 0, 2, 1, 0, 2]`
- Hamiltonian weight w_21: `-5.2719059284356143e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=7.2256884635241292e-01`
- fitted rem coeffs: `a_r=1.0511305135607701e-01`, `b_r=-1.3871557682379362e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963104e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963102e+00`, `y_exact=-4.1614673047138062e-01`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3196798592963102e+00`, `y_exact=-4.1614673047138062e-01`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3196798592963102e+00`
- target contribution to E_cdr_unmit: `1.4625909436474461e-03`
- target contribution to E_cdr_rem: `1.4625909436474468e-03`

### term 22
- pauli term from int row: `(5.2719059284356143e-03)*X(q(0))*Y(q(2))*Y(q(3))*X(q(5))`
- int observable row: `[1, 0, 2, 2, 0, 1]`
- Hamiltonian weight w_22: `5.2719059284356143e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.2774311536475871e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.5971110129438977e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3196798592963108e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3196798592963108e+00`, `y_exact=-4.1614673047138062e-01`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3196798592963108e+00`, `y_exact=-4.1614673047138062e-01`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.3196798592963108e+00`
- target contribution to E_cdr_unmit: `-1.4625909436474461e-03`
- target contribution to E_cdr_rem: `-1.4625909436474451e-03`

### term 23
- pauli term from int row: `(4.7435147736005108e-02)*Z(q(0))*Z(q(1))`
- int observable row: `[3, 3, 0, 0, 0, 0]`
- Hamiltonian weight w_23: `4.7435147736005108e-02`
- OGM effective shots used for this term: `4562`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0354703679224498e-02`, `x_rem=1.1463708325182682e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-2.3746701846965697e-02`, `x_rem=-2.6290010037165908e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.5789473684210527e-02`, `x_rem=-1.7480550533483987e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.1545979564637938e-01`, `x_r_target=2.3853587022960956e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 24
- pauli term from int row: `(5.3161545522654612e-02)*Z(q(0))*Z(q(4))`
- int observable row: `[3, 0, 0, 0, 3, 0]`
- Hamiltonian weight w_24: `5.3161545522654612e-02`
- OGM effective shots used for this term: `3726`
- fitted unmit coeffs: `a_u=-1.1025509577139509e-15`, `b_u=-4.0419825125745035e-18`
- fitted rem coeffs: `a_r=-9.6170051388637328e-16`, `b_r=-4.0419825125744588e-18`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.5583019881783988e-02`, `x_rem=1.7865305515232844e-02`, `y_exact=-5.5511151231257827e-17`
  - train[1] t_remaining=1: `x_unmit=1.9468055936386071e-02`, `x_rem=2.2319343088161556e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=4.2987641053197209e-03`, `x_rem=4.9283601421332536e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=6.1056105610561059e-02`, `x_r_target=6.9998369287719589e-02`
- target contribution to E_cdr_unmit: `-3.7935786617558771e-18`
- target contribution to E_cdr_rem: `-3.7935786617558663e-18`

### term 25
- pauli term from int row: `(-3.8063028881712008e-03)*Y(q(0))*Y(q(1))*Z(q(3))*Z(q(4))`
- int observable row: `[2, 2, 0, 3, 3, 0]`
- Hamiltonian weight w_25: `-3.8063028881712008e-03`
- OGM effective shots used for this term: `576`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-2.2146507666098807e-02`, `x_rem=-2.8713120961134431e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.7301038062283738e-02`, `x_rem=-2.2430931599928008e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-6.0658578856152515e-02`, `x_rem=-7.8644323443075148e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=0.0000000000000000e+00`, `x_r_target=-4.3824593077308808e-18`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 26
- pauli term from int row: `(-3.8063028881712008e-03)*X(q(0))*X(q(1))*Z(q(3))*Z(q(4))`
- int observable row: `[1, 1, 0, 3, 3, 0]`
- Hamiltonian weight w_26: `-3.8063028881712008e-03`
- OGM effective shots used for this term: `599`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-9.3457943925233638e-03`, `x_rem=-1.2116895761456428e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.0541516245487361e-02`, `x_rem=6.5527472385060445e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=5.4421768707482991e-02`, `x_rem=7.0558250148344964e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.5332197614991482e-02`, `x_r_target=1.9878314511554595e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 27
- pauli term from int row: `(5.1256723004769753e-03)*Y(q(0))*Z(q(1))*X(q(2))*Z(q(3))*X(q(4))*Y(q(5))`
- int observable row: `[2, 3, 1, 3, 1, 2]`
- Hamiltonian weight w_27: `5.1256723004769753e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.5155090949116965e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 28
- pauli term from int row: `(-5.1256723004769753e-03)*Y(q(0))*Z(q(1))*X(q(2))*Z(q(3))*Y(q(4))*X(q(5))`
- int observable row: `[2, 3, 1, 3, 2, 1]`
- Hamiltonian weight w_28: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116967e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116967e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.5155090949116972e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 29
- pauli term from int row: `(-5.1256723004769753e-03)*X(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*X(q(4))*Y(q(5))`
- int observable row: `[1, 3, 2, 3, 1, 2]`
- Hamiltonian weight w_29: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.5155090949116965e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.5155090949116965e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 30
- pauli term from int row: `(5.1256723004769753e-03)*X(q(0))*Z(q(1))*Y(q(2))*Z(q(3))*Y(q(4))*X(q(5))`
- int observable row: `[1, 3, 2, 3, 2, 1]`
- Hamiltonian weight w_30: `5.1256723004769753e-03`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.5155090949116967e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116967e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116967e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.5155090949116967e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.5155090949116967e+00`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 31
- pauli term from int row: `(5.5376587099558762e-02)*Z(q(0))*Z(q(2))`
- int observable row: `[3, 0, 3, 0, 0, 0]`
- Hamiltonian weight w_31: `5.5376587099558762e-02`
- OGM effective shots used for this term: `4562`
- fitted unmit coeffs: `a_u=1.5387903568463510e+00`, `b_u=-2.0288168470299805e-02`
- fitted rem coeffs: `a_r=1.3869996443307999e+00`, `b_r=-2.0288168470299999e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.3439083498567966e-02`, `x_rem=1.4909832296623939e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=2.8935795954265614e-01`, `x_rem=3.2102476712299782e-01`, `y_exact=4.1614673047138062e-01`
  - train[2] t_remaining=1: `x_unmit=2.7763157894736840e-01`, `x_rem=3.0801507281305807e-01`, `y_exact=4.1614673047138062e-01`
- target x values: `x_u_target=3.0830741892492225e-01`, `x_r_target=3.4204802079437979e-01`
- target contribution to E_cdr_unmit: `2.5148297680570371e-02`
- target contribution to E_cdr_rem: `2.5148297680570374e-02`

### term 32
- pauli term from int row: `(1.4370460325369306e-03)*Y(q(0))*Y(q(1))*Z(q(2))*Z(q(3))`
- int observable row: `[2, 2, 3, 3, 0, 0]`
- Hamiltonian weight w_32: `1.4370460325369306e-03`
- OGM effective shots used for this term: `576`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.8739352640545145e-02`, `x_rem=2.3511157834769834e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=2.7681660899653980e-02`, `x_rem=3.4730543313016435e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-3.6395147313691506e-02`, `x_rem=-4.5662839550844145e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.6447368421052631e-02`, `x_r_target=-2.0635540743136789e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 33
- pauli term from int row: `(1.4370460325369306e-03)*X(q(0))*X(q(1))*Z(q(2))*Z(q(3))`
- int observable row: `[1, 1, 3, 3, 0, 0]`
- Hamiltonian weight w_33: `1.4370460325369306e-03`
- OGM effective shots used for this term: `599`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=3.5514018691588788e-02`, `x_rem=4.4557339563498402e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-5.0541516245487361e-02`, `x_rem=-6.3411452276382868e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.3605442176870748e-02`, `x_rem=-1.7069943907247875e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=6.6439522998296419e-02`, `x_r_target=8.3357741414184020e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 34
- pauli term from int row: `(6.0648493027994375e-02)*Z(q(0))*Z(q(5))`
- int observable row: `[3, 0, 0, 0, 0, 3]`
- Hamiltonian weight w_34: `6.0648493027994375e-02`
- OGM effective shots used for this term: `4562`
- fitted unmit coeffs: `a_u=1.9391532523006474e-02`, `b_u=9.8863591647449534e-01`
- fitted rem coeffs: `a_r=1.6652349794355363e-02`, `b_r=9.8863591647449556e-01`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=5.7876184181537782e-01`, `x_rem=6.7396368784193894e-01`, `y_exact=9.9982691755849573e-01`
  - train[1] t_remaining=1: `x_unmit=5.8751099384344763e-01`, `x_rem=6.8415200770047024e-01`, `y_exact=9.9999993586766500e-01`
  - train[2] t_remaining=1: `x_unmit=5.8289473684210524e-01`, `x_rem=6.7877641213098305e-01`, `y_exact=9.9999993586766500e-01`
- target x values: `x_u_target=5.7752110173256332e-01`, `x_r_target=6.7251885561311731e-01`
- target contribution to E_cdr_unmit: `6.0638482127035494e-02`
- target contribution to E_cdr_rem: `6.0638482127035487e-02`

### term 35
- pauli term from int row: `(-3.6886262679400451e-03)*Y(q(0))*Y(q(1))*Z(q(3))*Z(q(5))`
- int observable row: `[2, 2, 0, 3, 0, 3]`
- Hamiltonian weight w_35: `-3.6886262679400451e-03`
- OGM effective shots used for this term: `576`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-6.9846678023850084e-02`, `x_rem=-9.1981119319022100e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.1141868512110725e-02`, `x_rem=4.1010739586664985e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=4.6793760831889082e-02`, `x_rem=6.1622723087761788e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-6.5789473684210523e-02`, `x_r_target=-8.6638185237910725e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 36
- pauli term from int row: `(-3.6886262679400451e-03)*X(q(0))*X(q(1))*Z(q(3))*Z(q(5))`
- int observable row: `[1, 1, 0, 3, 0, 3]`
- Hamiltonian weight w_36: `-3.6886262679400451e-03`
- OGM effective shots used for this term: `599`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-5.7943925233644861e-02`, `x_rem=-7.6306379222623399e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-2.5270758122743681e-02`, `x_rem=-3.3279071874778722e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-7.8231292517006806e-02`, `x_rem=-1.0302282162984214e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=8.0068143100511080e-02`, `x_r_target=1.0544177092668382e-01`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 37
- pauli term from int row: `(5.3161545522654612e-02)*Z(q(1))*Z(q(3))`
- int observable row: `[0, 3, 0, 3, 0, 0]`
- Hamiltonian weight w_37: `5.3161545522654612e-02`
- OGM effective shots used for this term: `3742`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-2.4298056155507560e-02`, `x_rem=-2.7478173297044276e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0117582718074924e-02`, `x_rem=-1.1441766760894858e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.5269220466113046e-02`, `x_rem=-1.7267648218169375e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.6344846342126734e-01`, `x_r_target=1.8484051457783657e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 38
- pauli term from int row: `(-3.8063028881712008e-03)*Y(q(3))*Y(q(4))`
- int observable row: `[0, 0, 0, 2, 2, 0]`
- Hamiltonian weight w_38: `-3.8063028881712008e-03`
- OGM effective shots used for this term: `1363`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=9.2923516797712644e-03`, `x_rem=1.0882115594061470e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-2.1632937892533146e-02`, `x_rem=-2.5333966997630145e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=2.1707670043415339e-02`, `x_rem=2.5421484553198256e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=4.2313117066290554e-03`, `x_r_target=4.9552174404118498e-03`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 39
- pauli term from int row: `(-3.8063028881712008e-03)*X(q(3))*X(q(4))`
- int observable row: `[0, 0, 0, 1, 1, 0]`
- Hamiltonian weight w_39: `-3.8063028881712008e-03`
- OGM effective shots used for this term: `1364`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-4.2796005706134094e-02`, `x_rem=-5.0117677107731816e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.9943019943019943e-02`, `x_rem=2.3354932722472941e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=7.2621641249092229e-04`, `x_rem=8.5045973499201736e-04`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.6200873362445413e-02`, `x_r_target=3.0683398823161567e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 40
- pauli term from int row: `(-5.1256723004769753e-03)*X(q(1))*X(q(2))*Y(q(3))*Y(q(5))`
- int observable row: `[0, 1, 1, 2, 0, 2]`
- Hamiltonian weight w_40: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `324`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.3595166163141995e-01`, `x_rem=-1.7971496568222492e-01`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=5.9523809523809521e-03`, `x_rem=7.8684727038116752e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=3.1347962382445140e-03`, `x_rem=4.1438978502832844e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-6.4102564102564097e-02`, `x_r_target=-8.4737398348741375e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 41
- pauli term from int row: `(-5.1256723004769753e-03)*Y(q(1))*Y(q(2))*Y(q(3))*Y(q(5))`
- int observable row: `[0, 2, 2, 2, 0, 2]`
- Hamiltonian weight w_41: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `311`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-2.9940119760479042e-02`, `x_rem=-3.9577946534142684e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=6.0240963855421690e-03`, `x_rem=7.9632735797612476e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-4.8859934853420196e-02`, `x_rem=-6.4588114702298000e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=3.4700315457413249e-02`, `x_r_target=4.5870465478372320e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 42
- pauli term from int row: `(-5.1256723004769753e-03)*X(q(1))*X(q(2))*X(q(3))*X(q(5))`
- int observable row: `[0, 1, 1, 1, 0, 1]`
- Hamiltonian weight w_42: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `321`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-2.2801302931596091e-02`, `x_rem=-3.0141120194405721e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=2.0833333333333332e-02`, `x_rem=2.7539654463340953e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=9.0322580645161285e-02`, `x_rem=1.1939772773783948e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=2.6315789473684209e-02`, `x_r_target=3.4786931953693805e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 43
- pauli term from int row: `(-5.1256723004769753e-03)*Y(q(1))*Y(q(2))*X(q(3))*X(q(5))`
- int observable row: `[0, 2, 2, 1, 0, 1]`
- Hamiltonian weight w_43: `-5.1256723004769753e-03`
- OGM effective shots used for this term: `312`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-4.4776119402985072e-02`, `x_rem=-5.9189705115240250e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-3.9735099337748346e-02`, `x_rem=-5.2525963479749631e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=8.3076923076923076e-02`, `x_rem=1.0981966825996883e-01`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=9.4637223974763408e-03`, `x_r_target=1.2510126948646999e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 44
- pauli term from int row: `(4.7435147736005108e-02)*Z(q(3))*Z(q(4))`
- int observable row: `[0, 0, 0, 3, 3, 0]`
- Hamiltonian weight w_44: `4.7435147736005108e-02`
- OGM effective shots used for this term: `4597`
- fitted unmit coeffs: `a_u=-2.6322348386608211e-15`, `b_u=-1.5998387020989821e-17`
- fitted rem coeffs: `a_r=-2.2476927039747951e-15`, `b_r=-1.5998387020989803e-17`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.1091393078970719e-02`, `x_rem=1.2988942492058564e-02`, `y_exact=-5.5511151231257827e-17`
  - train[1] t_remaining=1: `x_unmit=1.3416815742397137e-03`, `x_rem=1.5712205569105549e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-9.5777100565955595e-03`, `x_rem=-1.1216293953786673e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=7.6110866695996482e-02`, `x_r_target=8.9132146295438613e-02`
- target contribution to E_cdr_unmit: `-1.0262122799286385e-17`
- target contribution to E_cdr_rem: `-1.0262122799286379e-17`

### term 45
- pauli term from int row: `(6.0648493027994375e-02)*Z(q(2))*Z(q(3))`
- int observable row: `[0, 0, 3, 3, 0, 0]`
- Hamiltonian weight w_45: `6.0648493027994375e-02`
- OGM effective shots used for this term: `4597`
- fitted unmit coeffs: `a_u=1.6999959138570444e+00`, `b_u=2.2299356779776246e-02`
- fitted rem coeffs: `a_r=1.5000850983665348e+00`, `b_r=2.2299356779776246e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-8.4294587400177458e-03`, `x_rem=-9.5528216563586423e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=2.0572450805008943e-01`, `x_rem=2.3314065545096732e-01`, `y_exact=4.1614673047138062e-01`
  - train[2] t_remaining=1: `x_unmit=2.5293861558554637e-01`, `x_rem=2.8664681318434143e-01`, `y_exact=4.1614673047138062e-01`
- target x values: `x_u_target=2.3933128024637043e-01`, `x_r_target=2.7122607838718160e-01`
- target contribution to E_cdr_unmit: `2.6028001591874114e-02`
- target contribution to E_cdr_rem: `2.6028001591874107e-02`

### term 46
- pauli term from int row: `(-3.6886262679400451e-03)*Z(q(1))*Z(q(2))*Y(q(3))*Y(q(4))`
- int observable row: `[0, 3, 3, 2, 2, 0]`
- Hamiltonian weight w_46: `-3.6886262679400451e-03`
- OGM effective shots used for this term: `562`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-5.8823529411764705e-02`, `x_rem=-7.6554904251472336e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=7.6666666666666661e-02`, `x_rem=9.9776558541085594e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=2.7027027027027029e-02`, `x_rem=3.5173874926352142e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=6.4846416382252553e-02`, `x_r_target=8.4393290352305639e-02`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 47
- pauli term from int row: `(-3.6886262679400451e-03)*Z(q(1))*Z(q(2))*X(q(3))*X(q(4))`
- int observable row: `[0, 3, 3, 1, 1, 0]`
- Hamiltonian weight w_47: `-3.6886262679400451e-03`
- OGM effective shots used for this term: `578`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-4.0000000000000001e-02`, `x_rem=-5.2057334891001165e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.6184210526315791e-02`, `x_rem=4.7091339128372788e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=2.4305555555555556e-02`, `x_rem=3.1632061131684742e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-5.3097345132743362e-03`, `x_r_target=-6.9102656934957240e-03`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-0.0000000000000000e+00`

### term 48
- pauli term from int row: `(5.5376587099558762e-02)*Z(q(3))*Z(q(5))`
- int observable row: `[0, 0, 0, 3, 0, 3]`
- Hamiltonian weight w_48: `5.5376587099558762e-02`
- OGM effective shots used for this term: `4597`
- fitted unmit coeffs: `a_u=-1.2769346238863702e-02`, `b_u=1.0081873090447535e+00`
- fitted rem coeffs: `a_r=-1.0735027978473652e-02`, `b_r=1.0081873090447515e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=6.5128660159716056e-01`, `x_rem=7.7470726049355676e-01`, `y_exact=9.9982691755849573e-01`
  - train[1] t_remaining=1: `x_unmit=6.4669051878354200e-01`, `x_rem=7.6924020694630413e-01`, `y_exact=9.9999993586766500e-01`
  - train[2] t_remaining=1: `x_unmit=6.3909447104919459e-01`, `x_rem=7.6020468661405183e-01`, `y_exact=9.9999993586766500e-01`
- target x values: `x_u_target=6.5068191816981957e-01`, `x_r_target=7.7398798784106804e-01`
- target contribution to E_cdr_unmit: `5.5369860302860702e-02`
- target contribution to E_cdr_rem: `5.5369860302860702e-02`

### term 49
- pauli term from int row: `(1.4370460325369306e-03)*Z(q(1))*Y(q(3))*Y(q(4))*Z(q(5))`
- int observable row: `[0, 3, 0, 2, 2, 3]`
- Hamiltonian weight w_49: `1.4370460325369306e-03`
- OGM effective shots used for this term: `562`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-6.5743944636678195e-02`, `x_rem=-8.9807208821262530e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-4.0000000000000001e-02`, `x_rem=-5.4640596524936573e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=5.4054054054054057e-03`, `x_rem=7.3838643952616771e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=7.8498293515358364e-02`, `x_r_target=1.0722983959671850e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 50
- pauli term from int row: `(1.4370460325369306e-03)*Z(q(1))*X(q(3))*X(q(4))*Z(q(5))`
- int observable row: `[0, 3, 0, 1, 1, 3]`
- Hamiltonian weight w_50: `1.4370460325369306e-03`
- OGM effective shots used for this term: `578`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-8.6956521739130436e-03`, `x_rem=-1.1878390548899241e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=3.9473684210526314e-02`, `x_rem=5.3921641307503196e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-3.4722222222222220e-03`, `x_rem=-4.7431073372340799e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=8.8495575221238937e-03`, `x_r_target=1.2088627549764721e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 51
- pauli term from int row: `(8.1752954675092832e-02)*Z(q(1))*Z(q(4))`
- int observable row: `[0, 3, 0, 0, 3, 0]`
- Hamiltonian weight w_51: `8.1752954675092832e-02`
- OGM effective shots used for this term: `3666`
- fitted unmit coeffs: `a_u=1.9104738591001336e+00`, `b_u=3.9548473150278769e-02`
- fitted rem coeffs: `a_r=1.6636085405435630e+00`, `b_r=3.9548473150278769e-02`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.9640387275242047e-02`, `x_rem=-2.2554853234699627e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=1.8588496804667964e-01`, `x_rem=2.1346871189829991e-01`, `y_exact=4.1614673047138062e-01`
  - train[2] t_remaining=1: `x_unmit=2.0730046308907654e-01`, `x_rem=2.3806208375296697e-01`, `y_exact=4.1614673047138062e-01`
- target x values: `x_u_target=2.0367278797996660e-01`, `x_r_target=2.3389609259798150e-01`
- target contribution to E_cdr_unmit: `3.5044222400933002e-02`
- target contribution to E_cdr_rem: `3.5044222400933002e-02`

### term 52
- pauli term from int row: `(1.0346900732713493e-02)*Y(q(1))*X(q(2))*X(q(4))*Y(q(5))`
- int observable row: `[0, 2, 1, 0, 1, 2]`
- Hamiltonian weight w_52: `1.0346900732713493e-02`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-9.9380280793530951e-01`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.3361762641410107e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057010e+00`, `y_exact=1.8591576194071480e-02`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057010e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057010e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.3423734562057010e+00`
- target contribution to E_cdr_unmit: `6.4121731114912309e-05`
- target contribution to E_cdr_rem: `6.4121731114911157e-05`

### term 53
- pauli term from int row: `(-1.0346900732713493e-02)*Y(q(1))*X(q(2))*Y(q(4))*X(q(5))`
- int observable row: `[0, 2, 1, 0, 2, 1]`
- Hamiltonian weight w_53: `-1.0346900732713493e-02`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.3423734562057010e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057010e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057010e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057010e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.3423734562057010e+00`
- target contribution to E_cdr_unmit: `-2.0693801465426986e-02`
- target contribution to E_cdr_rem: `-2.7778809795179826e-02`

### term 54
- pauli term from int row: `(-1.0346900732713493e-02)*X(q(1))*Y(q(2))*X(q(4))*Y(q(5))`
- int observable row: `[0, 1, 2, 0, 1, 2]`
- Hamiltonian weight w_54: `-1.0346900732713493e-02`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=1.0000000000000000e+00`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=1.3423734562057019e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057019e+00`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057019e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.0000000000000000e+00`, `x_rem=-1.3423734562057019e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=-1.0000000000000000e+00`, `x_r_target=-1.3423734562057017e+00`
- target contribution to E_cdr_unmit: `-0.0000000000000000e+00`
- target contribution to E_cdr_rem: `-2.2974734853938846e-18`

### term 55
- pauli term from int row: `(1.0346900732713493e-02)*X(q(1))*Y(q(2))*Y(q(4))*X(q(5))`
- int observable row: `[0, 1, 2, 0, 2, 1]`
- Hamiltonian weight w_55: `1.0346900732713493e-02`
- OGM effective shots used for this term: `1`
- fitted unmit coeffs: `a_u=1.0000000000000000e+00`, `b_u=-9.9380280793530951e-01`
- fitted rem coeffs: `a_r=1.0000000000000000e+00`, `b_r=-1.3361762641410107e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057013e+00`, `y_exact=1.8591576194071480e-02`
  - train[1] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057013e+00`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=1.0000000000000000e+00`, `x_rem=1.3423734562057013e+00`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0000000000000000e+00`, `x_r_target=1.3423734562057015e+00`
- target contribution to E_cdr_unmit: `6.4121731114912309e-05`
- target contribution to E_cdr_rem: `6.4121731114915752e-05`

### term 56
- pauli term from int row: `(5.9449198989561303e-02)*Z(q(1))*Z(q(2))`
- int observable row: `[0, 3, 3, 0, 0, 0]`
- Hamiltonian weight w_56: `5.9449198989561303e-02`
- OGM effective shots used for this term: `4562`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=2.7539105529852389e-02`, `x_rem=3.0604414284333546e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.4951627088830254e-02`, `x_rem=-1.6615855193822554e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=6.5789473684210523e-03`, `x_rem=7.3112334966626133e-03`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.0484229231452688e-01`, `x_r_target=1.1651202487440317e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 57
- pauli term from int row: `(6.9796099722274796e-02)*Z(q(1))*Z(q(5))`
- int observable row: `[0, 3, 0, 0, 0, 3]`
- Hamiltonian weight w_57: `6.9796099722274796e-02`
- OGM effective shots used for this term: `4562`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-9.9140779907468599e-03`, `x_rem=-1.1564319308042071e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-1.5831134564643801e-02`, `x_rem=-1.8466295633844526e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=4.3859649122807018e-04`, `x_rem=5.1160278108384625e-04`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.8480675255442025e-01`, `x_r_target=2.1556863873940937e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 58
- pauli term from int row: `(6.9796099722274796e-02)*Z(q(2))*Z(q(4))`
- int observable row: `[0, 0, 3, 0, 3, 0]`
- Hamiltonian weight w_58: `6.9796099722274796e-02`
- OGM effective shots used for this term: `4597`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=-2.3070097604259095e-02`, `x_rem=-2.6549420385287393e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-4.1592128801431129e-02`, `x_rem=-4.7864856543318304e-02`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-2.0896821941663042e-02`, `x_rem=-2.4048381587396736e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=1.8829740431148262e-01`, `x_r_target=2.1669552640301937e-01`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 59
- pauli term from int row: `(5.9449198989561303e-02)*Z(q(4))*Z(q(5))`
- int observable row: `[0, 0, 0, 0, 3, 3]`
- Hamiltonian weight w_59: `5.9449198989561303e-02`
- OGM effective shots used for this term: `4597`
- fitted unmit coeffs: `a_u=0.0000000000000000e+00`, `b_u=0.0000000000000000e+00`
- fitted rem coeffs: `a_r=0.0000000000000000e+00`, `b_r=0.0000000000000000e+00`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=1.0204081632653060e-02`, `x_rem=1.2325741015414811e-02`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=-3.1305903398926647e-03`, `x_rem=-3.7815108839779543e-03`, `y_exact=0.0000000000000000e+00`
  - train[2] t_remaining=1: `x_unmit=-1.1754462342185459e-02`, `x_rem=-1.4198480943312081e-02`, `y_exact=0.0000000000000000e+00`
- target x values: `x_u_target=7.8750549934007916e-02`, `x_r_target=9.5124570564191518e-02`
- target contribution to E_cdr_unmit: `0.0000000000000000e+00`
- target contribution to E_cdr_rem: `0.0000000000000000e+00`

### term 60
- pauli term from int row: `(7.8236377789852360e-02)*Z(q(2))*Z(q(5))`
- int observable row: `[0, 0, 3, 0, 0, 3]`
- Hamiltonian weight w_60: `7.8236377789852360e-02`
- OGM effective shots used for this term: `6360`
- fitted unmit coeffs: `a_u=1.6519368593371655e+00`, `b_u=-9.0605290668406314e-03`
- fitted rem coeffs: `a_r=1.4132208821472505e+00`, `b_r=-9.0605290668403834e-03`
- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:
  - train[0] t_remaining=1: `x_unmit=5.9955822025875667e-03`, `x_rem=7.0083334875377908e-03`, `y_exact=0.0000000000000000e+00`
  - train[1] t_remaining=1: `x_unmit=2.4913494809688580e-01`, `x_rem=2.9121789021754302e-01`, `y_exact=4.1614673047138062e-01`
  - train[2] t_remaining=1: `x_unmit=2.6515270164447924e-01`, `x_rem=3.0994130268852516e-01`, `y_exact=4.1614673047138062e-01`
- target x values: `x_u_target=2.5466813117840892e-01`, `x_r_target=2.9768571785673043e-01`
- target contribution to E_cdr_unmit: `3.2204842616071608e-02`
- target contribution to E_cdr_rem: `3.2204842616071629e-02`

## Complete expanded energy expressions (all summing terms)
### E_cdr_unmit
`E_cdr_unmit = (-7.2476524580835413e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (6.0670030584275551e-02) + (1.4547697615567490e-01) + (-0.0000000000000000e+00) + (-5.1388310763032913e-17) + (1.1183806232021923e-01) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (1.1452795573299018e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-3.5487586922684906e-05) + (-5.2719052740607336e-03) + (5.2719055903359758e-03) + (1.4625909436474461e-03) + (-1.4625909436474461e-03) + (0.0000000000000000e+00) + (-3.7935786617558771e-18) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (2.5148297680570371e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (6.0638482127035494e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-1.0262122799286385e-17) + (2.6028001591874114e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (5.5369860302860702e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (3.5044222400933002e-02) + (6.4121731114912309e-05) + (-2.0693801465426986e-02) + (-0.0000000000000000e+00) + (6.4121731114912309e-05) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (3.2204842616071608e-02)`

### E_cdr_rem
`E_cdr_rem = (-7.2476524580835413e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (6.0670030584275565e-02) + (1.4547697615567490e-01) + (-0.0000000000000000e+00) + (-5.1388310763032870e-17) + (1.1183806232021924e-01) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (1.4848639324865926e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-3.5487586922684906e-05) + (-5.2719052740607328e-03) + (5.2719055903359775e-03) + (1.4625909436474468e-03) + (-1.4625909436474451e-03) + (0.0000000000000000e+00) + (-3.7935786617558663e-18) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (2.5148297680570374e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (6.0638482127035487e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (-1.0262122799286379e-17) + (2.6028001591874107e-02) + (-0.0000000000000000e+00) + (-0.0000000000000000e+00) + (5.5369860302860702e-02) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (3.5044222400933002e-02) + (6.4121731114911157e-05) + (-2.7778809795179826e-02) + (-2.2974734853938846e-18) + (6.4121731114915752e-05) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (0.0000000000000000e+00) + (3.2204842616071629e-02)`
