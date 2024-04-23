import sys
import numpy as np
import numpy.random as random

print("""
enter: 
    rudyfile: the rudy file your want to map into King's graph
    width: width of King's graph
    randseed: for sampling moves, spins, edges
    print: print the details or not, default not
    divideBy: divide the orignal weights by a value before mapping
example:
    python map_PSSA.py rudy_map/J11_02.rud 7 
    python map_PSSA.py rudy_map/J11_02.rud 7 0 print 100
""")

rudyfile=sys.argv[1]
L = int(sys.argv[2])
randseed = int(sys.argv[3]) 
random.seed(randseed)
op_print = sys.argv[4]
if op_print == "print":
    toPrint = True
else:
    toPrint = False
divideBy = float(sys.argv[5])
if divideBy == 0:
    divideBy = 1

### constant
N_complete = L+1 #9
T0 = 60.315/10
Ttmax_2 = 33.435/10
tmax = 7*10**7
multF = 1

### import input graph I(V,E) 
f = open(rudyfile, "r")
N, edge = [int(x) for x in next(f).split()]
E_I = set()
Jij_dict = dict()
Jij = np.zeros((N,N))
for cnt in range(edge):
    i, j, J = [int(x) for x in next(f).split()]
    J = J/divideBy
    if J != 0:
        ij = (i,j)
        E_I.add(tuple(sorted(ij)))
        Jij_dict[tuple(sorted(ij))] = J
        Jij[i-1][j-1] = J
        Jij[j-1][i-1] = J
f.close()

### function
def twoD2oneD(row,col): # (0,0), (0,1), ...
    return int(row*L+col+1)

def oneD2twoD(i): # 1, 2, ...
    row = int((i-1)/L)
    col = int((i-1)%L)
    return [row, col]

def copy_phi_KG(phi, KG):
    phi_copy = [[]] * len(phi)
    for i in range(len(phi)):
        phi_copy[i] = phi[i].copy()
    KG_copy = KG.copy()
    return (phi_copy, KG_copy)

def get_KG(phi):
    KG = np.zeros((L,L))
    for i in range(N):
        for idx in phi[i]:
            [row, col] = oneD2twoD(idx)
            KG[row, col] = int(i+1)
    return KG

def get_E_H(KG):
    E_H = set()
    for row in range(0,L):
        for col in range(0,L):
            phi_i = KG[row, col]
            for [row_n, col_n] in [[row, col+1], [row+1, col-1], [row+1, col], [row+1, col+1]]:
                if (0 <= row_n < L) and (0 <= col_n < L):
                    E_H.add(tuple(sorted( (phi_i, KG[row_n,col_n]) )))
    return E_H

def get_Eemb(E_H):
    intersect = E_I.intersection(E_H)
    return len(intersect)

def found_minor_embed(Eemb):
    return Eemb == len(E_I)

def T(t):
    # temperature
    return T0*(1-2*t/tmax)*(t<tmax/2) + Ttmax_2*(2-2*t/tmax)*(t>=tmax/2)

def ps(t): 
    # prob(shift)
    # ps(0) = 1, ps(tmax) = 0
    return 1-t/tmax 

def pa(t):
    # prob(allow_any_direction_shift)
    # pa(0) = 0.095, pa(tmax) = 0.487
    return 0.095 + (0.487-0.095)*t/tmax

def sample_i_u_for_shift(phi):
    i_option = []
    for idx in range(len(phi)):
        phi_i = phi[idx]
        if len(phi_i) > 1:
            i_option.append(idx+1)
    i = random.choice(i_option) # from 1 to N
    u = random.choice([phi[i-1][0], phi[i-1][-1]]) # from 1 to L^2
    return [i,u] 

def sample_j_v_for_shift(i, u, phi, KG, allow_any_direction_shift):
    [row, col] = oneD2twoD(u)
    v_option = []
    j_correspond = []
    if allow_any_direction_shift:
        # find all leaf v adjacent to u in H
        for [row_n, col_n] in [[row-1, col-1], [row-1, col], [row-1, col+1], [row, col-1], [row, col+1], [row+1, col-1], [row+1, col], [row+1, col+1]]: 
            if (0 <= row_n < L) and (0 <= col_n < L):
                v_n = twoD2oneD(row_n, col_n)
                j_n = int(KG[row_n, col_n])
                if j_n != i and (phi[j_n-1][0] == v_n or phi[j_n-1][-1] == v_n):
                    # get all neighbor that is leaf
                    v_option.append(v_n)
                    j_correspond.append(j_n)
    else:
        #follow guiding pattern
        i_complete = int(KG_complete[row, col])
        idx_u = phi_complete[i_complete-1].index(u)
        #find previous and next of u
        for idx_n in [idx_u-1, idx_u+1]:
            if 0 <= idx_n < len(phi_complete[i_complete-1]):
                v_n = phi_complete[i_complete-1][idx_n]
                [row_n, col_n] = oneD2twoD(v_n)
                j_n = int(KG[row_n, col_n])
                if j_n != i and (phi[j_n-1][0] == v_n or phi[j_n-1][-1] == v_n):
                    v_option.append(v_n)
                    j_correspond.append(j_n)
            
    if len(v_option) > 0:
        idx = random.choice(range(len(v_option)))
        v =  v_option[idx] # from 1 to L^2
        j = j_correspond[idx]
        success = True
    else:
        v = -1
        j = -1
        success = False

    return [j, v, success]

def shift(i, u, j, v, phi_orig, KG_orig):
    # delete u from phi(i), attach it to phi(j), next to v
    (phi, KG) = copy_phi_KG(phi_orig, KG_orig)
    idx_u = phi[i-1].index(u)
    phi[i-1].pop(idx_u)
    idx_v = phi[j-1].index(v)
    if idx_v == 0:
        phi[j-1].insert(0,u)
    else:
        phi[j-1].append(u)
    [row, col] = oneD2twoD(u)
    KG[row,col] = j
    return (phi, KG)

def sample_j_for_swap(i, k, phi, KG):
    j_option_set = set()
    for idx in phi[k-1]:
        [row, col] = oneD2twoD(idx)
        for [row_n, col_n] in [[row-1, col-1], [row-1, col], [row-1, col+1], [row, col-1], [row, col+1], [row+1, col-1], [row+1, col], [row+1, col+1]]: 
            if (0 <= row_n < L) and (0 <= col_n < L):
                j_n = int(KG[row_n, col_n])
                if j_n != k and j_n != i: 
                    j_option_set.add(j_n)
    j = random.choice(list(j_option_set))
    return j

def swap(i, j, phi_orig, KG_orig):
    # swap phi(i) with phi(j)
    (phi, KG) = copy_phi_KG(phi_orig, KG_orig)
    phi_tmp = phi[i-1]
    phi[i-1] = phi[j-1]
    phi[j-1] = phi_tmp
    KG[KG==i] = -1
    KG[KG==j] = i
    KG[KG==-1] = j
    return (phi, KG)

### assign index for each spin in King's Graph LxL
### starting from 1 to L^2
### -> KG_idx
KG_idx = np.zeros((L,L))
for i in range(0,L):
    for j in range(0,L):
        KG_idx[i,j] = twoD2oneD(i,j)
if toPrint:
#if True:
    print("Denote the spin number in King's graph as below")
    print(KG_idx)
    

### 1 ===== prepare initial placement of super vertices START ===== #

### step 1-a: prepare best known complete graph with N_complete phi in KG_LxL
### -> KG_complete, phi_complete

KG_complete = np.zeros((L,L))
phi_complete = [[]] *N_complete

KG_complete[0:L,0:1] = np.ones((L,1)) # spin 1
phi_complete[1-1] = [twoD2oneD(row,0) for row in range(0,L)]

for i in range(2,N_complete+1,2): # spins 2, 4, 6, ... goes down
    phi_i = []
    for k in range(1,L): # 1, ..., L-1
        col = k
        row = i-2+k-1
        if row >= L:
            row = L - (row-L) -1
        KG_complete[row,col] = int(i)
        phi_i.append(twoD2oneD(row,col))
    phi_complete[i-1] = phi_i

for i in range(3,N_complete+1,2): # spins 3, 5, 7, ... goes up
    phi_i = []
    for k in range(1,L): # 1, ..., L-1
        col = k
        row = i-2-k+1
        if row < 0:
            row = -row-1
        KG_complete[row,col] = int(i)
        phi_i.append(twoD2oneD(row,col))
    phi_complete[i-1] = phi_i

if toPrint:
    print("The minor embedding of complete graph K_"+str(N_complete)+" into King's graph KG_"+str(L)+","+str(L)+" is")
    print(KG_complete)
    print("The super vertex set [phi(1), phi(2), ..., phi(N_complete)] is")
    for i in range(0,N_complete):
        print("phi(%s): %s" % (i+1, phi_complete[i]))
    

### step 1-b: split spins to make the number of phi = N
### -> KG_init, phi_init
quo = int(np.floor(N / N_complete))
rem = int(np.floor(N % N_complete))
phi_init = [[]] *N
for i in range(1,N_complete+1):
    i_split_into = quo + 1*(i<= rem)
    i_len = len(phi_complete[i-1])
    i_quo = int(np.floor(i_len / i_split_into))
    i_rem = int(np.floor(i_len % i_split_into))
    last_ele_idx = -1
    for j in range(1, i_split_into+1):
        idx = i + (j-1)*N_complete
        num_ele = i_quo + 1*(j > (i_split_into - i_rem))
        phi_init[idx-1] = [phi_complete[i-1][k] for k in range(last_ele_idx+1, num_ele+last_ele_idx+1)]
        last_ele_idx = last_ele_idx + num_ele

# from new phi to KG
KG_init = get_KG(phi_init)

if toPrint:
    print("The init supervertex placement for K_"+str(N)+" in King's graph KG_"+str(L)+","+str(L)+" is")
    print(KG_init)

    print("The super vertex set [phi(1), phi(2), ..., phi(N)] is")
    for i in range(0,N):
        print("phi(%s): %s" % (i+1, phi_init[i]))

(phi, KG) = copy_phi_KG(phi_init, KG_init)
### 1 ===== prepare initial placement of super vertices END ===== #

### 2 ===== START =====
E_H = get_E_H(KG)
Eemb_phi = get_Eemb(E_H)

(phi_best, KG_best) = copy_phi_KG(phi, KG)
# get E(H) and calculate Emb(phi)
E_H_best = E_H
Eemb_best = Eemb_phi
### 2 ===== END =====

### 3 ===== START =====

# exchange phi
# update KG, only exchange phi(i) and phi(j)

t = 0
while not found_minor_embed(Eemb_best):
    if(t >= tmax):
        if toPrint:
            print("- Reach tmax")
        print("MINOR NOT FOUND")
        exit()

    if toPrint:
        print("t: %s" % t)
        print()

    # sample move
    move = random.choice(['shift', 'swap'], p=[ps(t), 1-ps(t)])
    if toPrint:
        print("- Sample move")
        print("    p(shift) = %s, p(swap) = %s" % (ps(t), 1-ps(t)))
        print("    move: %s" % move)
        print()

    if move == 'shift':
        # sample i and u
        [shift_i, shift_u] = sample_i_u_for_shift(phi)
        if toPrint:
            print("- Randomly select i from V(I) and u in Leaf[phi(i)]")
            print("    phi(i): %s" % shift_i)
            print("    leaf u to shift: %s" % shift_u)
            print()

        # sample allow_any_direction_shift 
        allow_any_direction_shift = random.choice([True, False], p=[pa(t), 1-pa(t)])
        if toPrint:
            print("- Sample allow_any_direction_shift")
            print("    p(allow_any_direction_shift) = %s" % pa(t))
            print("    allow_any_direction_shift: %s" % allow_any_direction_shift)
            print()

        [shift_j, shift_v, success] = sample_j_v_for_shift(shift_i, shift_u, phi, KG, allow_any_direction_shift)

        if success:
            if toPrint:
                print("- Randomly select v")
                print("    phi(j): %s" % shift_j)
                print("    leaf v to shift: %s" % shift_v)
                print()
        else:
            if toPrint:
                print("A valid neighboring v does not exist, skip this proposal")
            continue

        (phi_proposed, KG_proposed) = shift(shift_i, shift_u, shift_j, shift_v, phi, KG)


    else: # if move == 'swap'
        # swap phi(i) and phi(j), where phi(j) is adjacent to phi(k)
        swap_i = random.choice(range(1,N+1))
        swap_k = random.choice(range(1,N+1))
        while swap_i == swap_k:
            swap_k = random.choice(range(1,N+1))
        if toPrint:
            print("- Randomly select i, k from V(I)")
            print("    phi(i): %s" % swap_i)
            print("    phi(k): %s" % swap_k)
            print()
        swap_j = sample_j_for_swap(swap_i, swap_k, phi, KG)
        if toPrint:
            print("- Randomly select j")
            print("    phi(j): %s" % swap_j)
            print()
            print("  -> swap phi(i) and phi(j)")
            print()

        (phi_proposed, KG_proposed) = swap(swap_i, swap_j, phi, KG)

    if toPrint:
        print("The proposed supervertex placement is")
        print(KG_proposed)
    
        print("The super vertex set [phi(1), phi(2), ..., phi(N)] is")
        for i in range(0,N):
            print("phi(%s): %s" % (i+1, phi_proposed[i]))

    # 17 ==== 
    Eemb_proposed = get_Eemb(get_E_H(KG_proposed))
    Eemb_diff = Eemb_proposed - Eemb_phi
    if toPrint:
        print()
        print("- Evaluate acceptance of proposed move")
        print("    Eemb(phi_proposed) = %s" % Eemb_proposed)
        print("    E_diff = Eemb(phi_proposed) - Eemb(phi): %s" % Eemb_diff)
    Tt = T(t)
    if toPrint:
        print("    Temperature: %s" % Tt)
        print()

    # update or not
    compare = np.exp(Eemb_diff/Tt)
    if toPrint:
        print("    exp(E_diff/T(t)): %s" % compare)
    rndf = random.random()
    if toPrint:
        print("    random_float: %s" % rndf)
        print()
    if compare > rndf:
        if toPrint:
            print("- Accept and Update")
            print("    update phi <- phi_proposed")
            print()
        phi = phi_proposed
        KG = KG_proposed
        Eemb_phi = Eemb_proposed
        if Eemb_phi > Eemb_best:
            if toPrint:
                print("- Eemb(phi) > Eemb(phi_best)")
                print("    update phi_best <- phi")
                print()
            (phi_best, KG_best) = copy_phi_KG(phi, KG)
            Eemb_best = Eemb_phi
            if found_minor_embed(Eemb_best):
                break
    else:
        if toPrint:
            print("- Not accept, phi unchange")
            print()

    print(".", end="")
    if t%100 == 0:
        print(" ", end="")
    t = t + 1

if toPrint:
    print("MINOR FOUND at time %s" % t)
    print("- Eemb(phi_best) == |E(I)|")
    print("    at time %s, minor found" % t)
    print()
    print("The minor embedding is")
    print(KG_best)
    
    print("The super vertex set [phi(1), phi(2), ..., phi(N)] is")
    for i in range(0,N):
        print("phi(%s): %s" % (i+1, phi_best[i]))


# ======================================================================================================
# ===== Jij mapping START =====
if toPrint:
    print("======================================================================================================")

# calculate minimum coupling strength required within each phi(i) for i in V(I)
Fii = np.zeros((N,1))
for i in range(0,N):
    # require Fii > sum(abs(Jij[0:N, i:i+1]))
    # for now, set Fii = 1.1 * sum(abs(Jij[0:N, i:i+1]))
    Fii[i] = multF * sum(abs(Jij[0:N, i:i+1]))
if toPrint:
    print("Calculate the coupling strength needed inside each phi(i)")
    print(Fii)

# if Fii[i] = 0 => still need to make coupling in phi(i)
Fii = Fii + 1*(Fii==0)
if toPrint:
    print("Adding coupling strength within Fii[i] == 0")
    print(Fii)


Jij_KG = np.zeros((L**2,L**2))
Jij_rudy = []
# (1) add coupling inside each phi(i)
for i in range(0,N):
    for k in range(0,len(phi[i])-1):
        s1 = phi_best[i][k]
        s2 = phi_best[i][k+1]
        Jij_i = Fii[i][0]
        if Jij_i != 0:
            Jij_KG[s1-1, s2-1] = Jij_i 
            Jij_KG[s2-1, s1-1] = Jij_i
            Jij_rudy.append(str(s1) + " " + str(s2) + " " + str(Jij_i) + "\n")

# (2) from the list of original Jij find all the edges one by one
for row in range(0,L):
    for col in range(0,L):
        s1 = twoD2oneD(row, col)
        i = int(KG_best[row][col])
        for [row_n, col_n] in [[row, col+1], [row+1, col-1], [row+1, col], [row+1, col+1]]: 
            if (0 <= row_n < L) and (0 <= col_n < L):
                s2 = twoD2oneD(row_n, col_n)
                j = int(KG_best[row_n, col_n])
                ij = (i,j)
                if Jij_dict.get(tuple(sorted(ij))) != None:
                    Jij_i = Jij_dict.pop(tuple(sorted(ij)))
                    Jij_KG[s1-1, s2-1] = Jij_i 
                    Jij_KG[s2-1, s1-1] = Jij_i
                    Jij_rudy.append(str(s1) + " " + str(s2) + " " + str(Jij_i) + "\n")

# ===== Jij mapping END =====

# output rudyfile of King's graph
rudyname = rudyfile.split('/')[-1].split('.')[0]
output_rudyfile = "output_map/"+rudyname+"_KG_"+str(L)+"_"+str(L)+"_r"+str(randseed)+".rud"
f = open(output_rudyfile, "w")
f.write(str(L**2) + " " + str(len(Jij_rudy)) + "\n")
f.writelines(Jij_rudy)
f.close()
print("write to %s" % output_rudyfile)
