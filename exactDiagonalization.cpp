/* Exact Diagonalization Lanczos version
   In this state the convention is defined as c1_u^dagger*c2_u^dagger*c3_u^dagger...c1_d^dagger*c2d^dagger...|0>
   Using this convention the hopping term is all postive for nearest neighbor.
   Author: Tsung-Han Lee                      
*/
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <ctime>
#include "matrix.h"
#include "myutil.h"
#include "random.h"
#include "diag.h"
#include <omp.h>
#include <cstdlib>
#include <cassert>
#include <complex>

using namespace std;

typedef enum
{
	anderson = 0, hubbard = 1
} Model;

typedef struct Parameters
{
    int N;// length of the chain
    int omegaPoints;// gridpoints for the frequency of A(omega) and G(omega)
    double u;// Coulomb interaction
    double eps;// on-site energy
    double t;// hopping strength
    double broadening;// broadening parameter for the delta-peaks
    double bandWidth;// range for the frequency
    Model model;// defines the model (either U_i = U*delta_{i,1} or U_i = U)
    int itmax;// maximum lanczos iterations
} Parameters;

typedef map<int, map<int, int> > QSzCount;// total number, Sz, number of state
//typedef map<int, map<int, vector<double> > > Energies;// total number, Sz, energies
typedef map<int, map<int, Matrix> > States;// total number, Sz, eigenstates
//typedef map<int, map<int, vector<double> > > States; // storing the states 
typedef map<int, map<int, vector<vector<int> > > > Basis;
// total number, Sz, basis |-1,0,2,1,1.....> as occupation -1 spin dn,1 spin up, 0 empty, 2 double occupied
//typedef vector<map<int, map<int, Matrix> > > Lanczos_States;// storing all the lanczos states

void buildbasis(const int N, vector<int> &s, Basis &basis, QSzCount &qszcount, map<int, int> &countSubspaces, ofstream &info );
bool newConfiguration(vector<int> &s, int lower, int upper);// generate new configuration from lower(-1) spin dn to upper(2) double occupied
//calculating ground state energy and ground state lanczos coefficient for wavfunction calculation
void lanczos( bool gf, Parameters parameters, Basis &basis, QSzCount &qszcount, vector<double> &a, vector<double> &b, States &init ,vector<double> &energies, vector<double> &gs_lanczos_coeff ,ofstream &info );
//calculating ground state wavfunction
States lanczos_gs_wavefn(Parameters parameters, Basis &basis, QSzCount &qszcount, vector<double> &energies, vector<double> &gs_lanczos_coeff ,ofstream &info );
//c_i^dagger|g.s.>
States c_i_d_u( Parameters parameters, Basis &basis, QSzCount &qszcount, int site, States &gs_wavefn );
//c_i|g.s.>
States c_i_u( Parameters parameters, Basis &basis, QSzCount &qszcount, int site, States &gs_wavefn );
States H_dot_u( Parameters parameters, Basis &basis, QSzCount &qszcount, States &u );// H|u>
double u_dot_up( Parameters parameters, Basis &basis, QSzCount &qszcount, States &u, States &up );// <u|up> u dot uprime
vector<complex<double> > greenfunction(int e_or_h, Parameters parameters, Basis &basis, QSzCount &qszcount, vector<double> &a, vector<double> &b, double b0 , double e0, vector<double> &omega);
complex<double> continuefraction(int e_or_h, complex<double> &z, vector<double> &a, vector<double> &b);
vector<double> diagonalize_tri(vector<double> diag , vector<double> offdiag , vector<double> &gs_lanczos_coeff );// diagonalize tridigonal matrix

int main(int argc, const char* argv[])
{
    int N=6, itmax=100 , charge , spin;
    //RanGSL rand(1234);//initial random number generator with seed
    Parameters parameters = {N, 5000, 2., -1., -0.5, 0.1, 10, hubbard/*anderson*/, itmax};

    if (argc==1) {
       cout << "commandline argument: N, U, mu, t, broaden, model(0 for anderson or 1 for hubbard), itmax " <<endl;
       return 1;
    }

    // reads in command line arguments, it is not essential
    switch (argc)
    {
        case 8:
            parameters.itmax =atoi(argv[7]);
        case 7:
            //parameters.model = (Model) atoi(argv[6]);
            parameters.model = static_cast<Model> (atoi( argv[6]) );
        case 6:
            parameters.broadening = atof(argv[5]);
        case 5:
            parameters.t = atof(argv[4]);
        case 4:
            parameters.eps = atof(argv[3]);
        case 3:
            parameters.u = atof(argv[2]);
        case 2:
            parameters.N = atoi(argv[1]);
    }

    if(parameters.model == anderson)
      clog << "start" << parameters.N << "-site Anderson chain:" << endl;
    else
      clog << "start" << parameters.N << "-site Hubbard chain:" << endl;

    time_t start, end;
    time(&start);
    ofstream info("info.dat");// print out the runtime information
    vector<double> energies;// storing energies
    //States u_n, u_n_p1, u_n_m1;//storing temporary lanczos basis
    States gs_wavefn;// storing ground state wave function
    States c_0_d_gswf, c_0_gswf;// storing state c_0^dagger|g.s.>
    //States states;// storing eigevstates, as a orthonormal matrix
    // s[i] labels a state at site i: s[i] = 0 means empty state,
    // -1 means spin down state, 1 means spin up state and 2 means doubly occupied state
    vector<int> s(parameters.N, -1);// initialize as all spin down
    //Matrix hamiltonian;// storing temporary hamiltonian for diagonalization
    // qszcount counts the number of subspaces in quantum numbers Q, Sz
    QSzCount qszcount;//count the size of subspace for each block
    Basis basis;// storing the basis for eahc block
    //Lanczos_States lanczos_states; // storing lanczos basis for constructing ground state wavefunction
    map<int, int> countSubspaces;// storing the number of subspace which has the same size. (first integer size, second interger occur times)
    vector<double> a, b; // storing the diagonal elements a and off diagonal elements b for tridiagonal matrix
    vector<double> gs_lanczos_coeff; // storing ground state lanczos coefficients
    vector<double> omega(parameters.omegaPoints);// frequency grid
    double gs_energy;//ground state energy
    char str[100];

    info << parameters.N << "-site chain\n" << endl;
    if (parameters.model == anderson)
        info << "tight-binding with interaction on first site(Anderson) only\n" << endl;
    else
        info << "Hubbard model\n" << endl;
    info << "U = " << parameters.u << "\neps = " << parameters.eps << "\nt = ";
    info << parameters.t << "\n" << endl;

    // initialize the size of space for each block as 0 
    for (int q = -parameters.N; q <= parameters.N; q++)// the real charge should plus N 
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
            qszcount[q][sz] = 0;
    
    // Set up the basis
    buildbasis(parameters.N, s, basis, qszcount, countSubspaces, info );
    // first lanczos to get ground state lanczos coefficient
    lanczos(false, parameters, basis, qszcount, a, b , c_0_d_gswf , energies, gs_lanczos_coeff, info );
    // calculating ground state wave function by retracing lacnzos basis |u_n>
    gs_wavefn = lanczos_gs_wavefn(parameters, basis, qszcount, energies, gs_lanczos_coeff ,info );
    gs_energy = energies[0];
    //check g.s. lanczos coefficient in lanczos basis
/*    double sum =0;
    for (int i=0; i<gs_lanczos_coeff.size(); i++) {
        sum += pow(gs_lanczos_coeff[i],2);
        clog << gs_lanczos_coeff[i] << endl;
    }
    clog << "sum of square of lanczos coefficeint:" << sum << endl;
*/
    //check g.s. wavefunction in original basis
/*    double sum =0;
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if ( qszcount[q][sz]==0 ) continue;
            for (int r=0; r< qszcount[q][sz]; r++) {
                sum += pow(gs_wavefn[q][sz].get(r,0),2);
                cout << gs_wavefn[q][sz].get(r,0) << endl;
            }
        }
    }
    clog << "sum of square of ground state coefficeint:" << sum << endl;
*/

    double stepWidth = 2*parameters.bandWidth*parameters.t/parameters.omegaPoints;
    for (int i = 0; i < parameters.omegaPoints; i++) {
        omega[i] = -parameters.bandWidth*parameters.t + i*stepWidth;
    }

    // Apply c_i^dagger|g.s.>
    c_0_d_gswf = c_i_d_u( parameters, basis, qszcount, 0 , gs_wavefn );
    //check electron wavefunction
/*    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if ( qszcount[q][sz]==0 ) continue;
            for (int r=0; r< qszcount[q][sz]; r++) {
                clog << c_0_d_gswf[q][sz].get(r,0) << endl;
            }
        }
    }
*/
    //calculate greens function coefficient
    double b0= sqrt(u_dot_up( parameters, basis, qszcount, c_0_d_gswf, c_0_d_gswf));
    clog << "norm of c_0_dagger|g.s.>=" << b0 << endl;

    lanczos(true, parameters, basis, qszcount, a, b, c_0_d_gswf , energies, gs_lanczos_coeff, info );
    vector<complex<double> > gf_e = greenfunction(1, parameters, basis, qszcount, a, b, b0 , gs_energy, omega);//1 for electron

    // Apply c_i|g.s>
    c_0_gswf = c_i_u( parameters, basis, qszcount, 0 , gs_wavefn ); 
    //check hole wavefunction
/*    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if ( qszcount[q][sz]==0 ) continue;
            for (int r=0; r< qszcount[q][sz]; r++) {
                clog << c_0_gswf[q][sz].get(r,0) << endl;
            }
        }
    }
*/
    b0= sqrt(u_dot_up( parameters, basis, qszcount, c_0_gswf, c_0_gswf));
    lanczos(true, parameters, basis, qszcount, a, b, c_0_gswf , energies, gs_lanczos_coeff, info );
    vector<complex<double> > gf_h = greenfunction(0, parameters, basis, qszcount, a, b, b0 , gs_energy, omega);//0 for hole

    if(parameters.model==anderson)
      sprintf(str,"greenFunction_anderson_N%dU%3.2f.dat",parameters.N,parameters.u);
    else
      sprintf(str,"greenFunction_hubbard_N%dU%3.2f.dat",parameters.N,parameters.u);
    ofstream printGF(str);

    for (int i = 0; i < parameters.omegaPoints; i++)
        printGF << omega[i] << "\t"  << real(gf_e[i]) + real(gf_h[i]) << "\t" << imag(gf_e[i]) + imag(gf_h[i]) << endl;
    printGF.close();


    return 0;
}

void buildbasis(int N,  vector<int> &s, Basis &basis, QSzCount &qszcount, map<int, int> &countSubspaces, ofstream &info )
{
    bool hasNext = true;
    for (;;)
    {
        int charge = 0;
        int spin = 0;
        // loop over all sites for looking the Q and Sz, start from state
        for (unsigned int i = 0; i < s.size(); i++)
        {
            // Calculate spin and charge of each configuration...
            charge += abs(s[i])-1;
            spin += s[i]*(2-abs(s[i]));// the term *(2-abs(s[i])) is set for double occupied state, i.e. spin=0.
        }
        // and store the configuration in the corresponding Q, Sz subspace
        basis[charge][spin].push_back(s);
        qszcount[charge][spin]++;
        // leave the loop if there exists no further basis state
        if (!hasNext)
            break;
        // determine next configuration and if there exists another one after that
        hasNext = newConfiguration(s, -1, 2);
    }

    // find and print largest subspace
    int largestSubspace = 0;
    for (int q = -N; q <= N; q++)
        for (int sz = -N; sz <= N; sz++)
        {
            if (qszcount[q][sz] > largestSubspace)
                largestSubspace = qszcount[q][sz];
            countSubspaces[qszcount[q][sz]]++;
        }

    // count number and sizes of subspaces
    for (map<int, int>::iterator it = countSubspaces.begin(); it != countSubspaces.end(); it++)
        info << (*it).second/*number of countSubspaces*/ << " subspaces of size " << (*it).first/*subsoace size (key qszcount)*/ << endl;
    info << "\nLargest subspace = " << largestSubspace << endl << endl;

}

bool newConfiguration(vector<int> &s, int lower, int upper)
{
    for (unsigned int i = 0; i < s.size(); i++)
    {
        if (s[i] < upper)
        {
            // increase one state, then leave the loop
            s[i]++;
            break;
        } else
            s[i] = lower;
    }
    // if there is any state not doubly occupied, we have some more
    // states to build and return true, ... 
    for (unsigned int i = 0; i < s.size(); i++)
        if (s[i] != upper)
            return true;
    // ... else we return false
    return false;
}

double u_dot_up( Parameters parameters, Basis &basis, QSzCount &qszcount, States &u, States &up )
{
    double sum=0;
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if (qszcount[q][sz] == 0) continue;
            sum += (u[q][sz].returnTransposed()*up[q][sz]).get(0,0);
        }
    }
    //cout << sum << endl;
    return sum;
};

States H_dot_u( Parameters parameters, Basis &basis, QSzCount &qszcount, States &u )
{
//    clog<<"enter H_dot_u"<<endl;
    //Matrix hamiltonian;//storing temporary partial hamiltonian for q and sz
    Matrix Hu_q_sz;
    States Hu;// storing H_dot_u

    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if (qszcount[q][sz] == 0)
                continue;

//            clog << "calculating H|u> : Q = " << q << ", Sz = " << sz << ", subspace size: " << qszcount[q][sz] << endl;
            // set up Hamiltonian in this subspace:
            //hamiltonian.resize(qszcount[q][sz], qszcount[q][sz]);
            //hamiltonian.zero();
            Hu_q_sz.resize(qszcount[q][sz],1);
            Hu[q][sz].resize(qszcount[q][sz],1);
            //set up u_q_sz
     
            #pragma omp parallel for
            for (int r = 0; r < qszcount[q][sz]; r++) // ket state
            {
                double sum=0;
                if (parameters.model == anderson)
                {
                    // Problem 2 a):
                    // =============
                    // Fill in the diagonal matrix elements of the Hamiltonian
                    // for the model with only first-site(Anderson) interaction
                    double temp=0; //temperory memory for diagonal element
                    //cout << basis[q][sz][r][m] << endl;
                    if ((basis[q][sz][r][0] == -1) )
                          temp += parameters.eps;
                    if ((basis[q][sz][r][0] ==  1) )
                          temp += parameters.eps;
                    if ((basis[q][sz][r][0] ==  0) )
                          temp =  temp;
                    if ((basis[q][sz][r][0] == 2) )
                          temp += 2*parameters.eps+parameters.u;

                    //hamiltonian.set(r,r, temp);
                    sum+=temp*u[q][sz].get(r,0);
                }
                else
                {
                    // Problem 2 b):
                    // =============
                    // Fill in the diagonal matrix elements of the Hamiltonian
                    // for the Hubbard model
                    double temp=0; //temperory memory for diagonal element
                    for(int m=0; m<parameters.N; m++)
                    { 
                      //cout << basis[q][sz][r][m] << endl;
                      if ((basis[q][sz][r][m] == -1) )
                            temp += parameters.eps;
                      if ((basis[q][sz][r][m] ==  1) )
                            temp += parameters.eps;
                      if ((basis[q][sz][r][m] ==  0) )
                            temp =  temp;
                      if ((basis[q][sz][r][m] == 2) )
                            temp += 2*parameters.eps+parameters.u;

                    }
                    //hamiltonian.set(r,r, temp);
                    sum+=temp*u[q][sz].get(r,0);
                    //cout << "diag element=" << hamiltonian.get(r,r) <<endl;
                                        
                }
                
                // hopping between sites:
                //#pragma omp parallel for shared(sum, hamiltonian,u)
                for (int rp = 0; rp < qszcount[q][sz]; rp++) // bra state
                {
                    for (int m = 0; m < parameters.N-1; m++)// searching hoping term from the basis
                    {
                        bool p = false;
                        for (int mp = 0; mp < parameters.N; mp++)
                        {
                            // if anything but two neighbouring sites...
                            if ((mp == m) || (mp == m+1))
                                continue;
                            // ... are different from each other... 
                            if (basis[q][sz][r][mp] != basis[q][sz][rp][mp])
                                p = true;
                        }//mp
                        // ... then there couldn't be a non-vanishing matrix element in the Hamiltonian
                        if (p)
                            continue;
                        
                        // Problem 2 c):
                        // ==========
                        // In the following, fill in all the missing matrix elements
                        
                        if ((basis[q][sz][r][m] == 0) && (basis[q][sz][r][m+1] == 1) && (basis[q][sz][rp][m] == 1) && (basis[q][sz][rp][m+1] == 0))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == -1) && (basis[q][sz][r][m+1] == 1) && (basis[q][sz][rp][m] == 2) && (basis[q][sz][rp][m+1] == 0))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 0) && (basis[q][sz][r][m+1] == -1) && (basis[q][sz][rp][m] == -1) && (basis[q][sz][rp][m+1] == 0))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 1) && (basis[q][sz][r][m+1] == -1) && (basis[q][sz][rp][m] == 2) && (basis[q][sz][rp][m+1] == 0))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 0) && (basis[q][sz][r][m+1] == 2) && (basis[q][sz][rp][m] == 1) && (basis[q][sz][rp][m+1] == -1))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == -1) && (basis[q][sz][r][m+1] == 2) && (basis[q][sz][rp][m] == 2) && (basis[q][sz][rp][m+1] == -1))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 0) && (basis[q][sz][r][m+1] == 2) &&(basis[q][sz][rp][m] == -1) && (basis[q][sz][rp][m+1] == 1))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 1) && (basis[q][sz][r][m+1] == 2) && (basis[q][sz][rp][m] == 2) && (basis[q][sz][rp][m+1] == 1))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 1) && (basis[q][sz][r][m+1] == 0) &&(basis[q][sz][rp][m] == 0) && (basis[q][sz][rp][m+1] == 1))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 1) && (basis[q][sz][r][m+1] == -1) &&(basis[q][sz][rp][m] == 0) && (basis[q][sz][rp][m+1] == 2))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == -1) && (basis[q][sz][r][m+1] == 0) &&(basis[q][sz][rp][m] == 0) && (basis[q][sz][rp][m+1] == -1))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == -1) && (basis[q][sz][r][m+1] == 1) &&(basis[q][sz][rp][m] == 0) && (basis[q][sz][rp][m+1] == 2))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 2) && (basis[q][sz][r][m+1] == 0) &&(basis[q][sz][rp][m] == -1) && (basis[q][sz][rp][m+1] == 1))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 2) && (basis[q][sz][r][m+1] == -1) && (basis[q][sz][rp][m] == -1) && (basis[q][sz][rp][m+1] == 2))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 2) && (basis[q][sz][r][m+1] == 0) && (basis[q][sz][rp][m] == 1) && (basis[q][sz][rp][m+1] == -1))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                        if ((basis[q][sz][r][m] == 2) && (basis[q][sz][r][m+1] == 1) && (basis[q][sz][rp][m] == 1) && (basis[q][sz][rp][m+1] == 2))
                            //hamiltonian.set(r, rp, parameters.t);
                            sum+=parameters.t*u[q][sz].get(rp,0);
                    }//m
                    //sum+=hamiltonian.get(r,rp)*u[q][sz].get(rp,0);
                }//rp
                Hu[q][sz].set(r,0,sum);
            }//r

            Hu_q_sz.erase();

        }//sz
    }//q
    return Hu;
}

void lanczos(bool gf, Parameters parameters, Basis &basis, QSzCount &qszcount , vector<double> &a, vector<double> &b, States &init ,vector<double> &energies, vector<double> &gs_lanczos_coeff , ofstream &info )
{
    RanGSL rand(1234);//initial random number generator with seed
    States u_n, u_n_p1, u_n_m1;//storing temporary lanczos basis
    a.clear();//clear diagonal element
    b.clear();//clear diagonal element
    energies.clear();//clear eigenenergies

    char str[40];
    if(!gf){
      if(parameters.model==anderson)
        sprintf(str,"energies_anderson_N%dU%3.2f.dat",parameters.N,parameters.u);
      else
        sprintf(str,"energies_hubbard_N%dU%3.2f.dat",parameters.N,parameters.u);
    }
    if(gf){
      if(parameters.model==anderson)
        sprintf(str,"energies_gf_anderson_N%dU%3.2f.dat",parameters.N,parameters.u);
      else
        sprintf(str,"energies_gf_hubbard_N%dU%3.2f.dat",parameters.N,parameters.u);
    }

    ofstream energiesOfStream(str);

    if(gf) clog <<"GF ";
    clog << "iteration=0:" << endl;
    // Set up the initial random Lanczos state u0    
    int hil_count = 0;
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if ( qszcount[q][sz]==0 ) continue;
            if(gf) {
               u_n[q][sz].resize(qszcount[q][sz],1);
               for ( int r=0; r<qszcount[q][sz]; r++){
                   u_n[q][sz].set(r,0,init[q][sz].get(r,0));
                   //cout << u_n[q][sz].get(r,0) << endl;
                   hil_count+=1;
               }
            }
            else {
               u_n[q][sz].resize(qszcount[q][sz],1);
               for (int r=0; r< qszcount[q][sz]; r++) {
                   u_n[q][sz].set(r,0,rand());
                   //cout << u_n[q][sz].get(r,0) << endl;
                   hil_count+=1;
               }
            }
        }
    }
    assert (pow(4.,parameters.N)==hil_count);
    clog << "total size of hilbert space:" << hil_count << endl;

    States Hu_n=H_dot_u( parameters, basis, qszcount, u_n ); 
    //calculate a_n=<u_n|H|u_n>/<u_n|u_n>
    a.push_back( u_dot_up( parameters, basis, qszcount, u_n, Hu_n) / u_dot_up( parameters, basis, qszcount, u_n, u_n) );
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        { 
            if (qszcount[q][sz]==0) continue; 
            Matrix a_u_n_q_sz = u_n[q][sz];
            a_u_n_q_sz.multiply( a[0] );
            u_n_p1[q][sz] = Hu_n[q][sz] - a_u_n_q_sz  ;
        }
    }

    if(gf) clog <<"GF ";
    clog << "iteration=1:" << endl;
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if (qszcount[q][sz]==0) continue;
            u_n_m1[q][sz] = u_n[q][sz];
            u_n[q][sz] = u_n_p1[q][sz];
        }
    }
    Hu_n=H_dot_u( parameters, basis, qszcount, u_n );
    //calculate a_n=<u_n|H|u_n>/<u_n|u_n>
    a.push_back( u_dot_up( parameters, basis, qszcount, u_n, Hu_n) / u_dot_up( parameters, basis, qszcount, u_n, u_n) );
    b.push_back( sqrt( u_dot_up( parameters, basis, qszcount, u_n, u_n) / u_dot_up( parameters, basis, qszcount, u_n_m1, u_n_m1) ) );
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if (qszcount[q][sz]==0) continue;
            Matrix a_u_n_q_sz = u_n[q][sz];
            Matrix b_u_nm1_q_sz = u_n_m1[q][sz];
            a_u_n_q_sz.multiply( a[1] );
            b_u_nm1_q_sz.multiply( pow(b[0], 2) );
            
            u_n_p1[q][sz] = Hu_n[q][sz] - a_u_n_q_sz - b_u_nm1_q_sz ;
        }
    }

    energies = diagonalize_tri( a, b, gs_lanczos_coeff );
    clog << "g.s. energy:" << energies[0] << endl;
    
    for (int it=2; it<parameters.itmax; it++)
    {
        double en, enm1=energies[0], de=pow(10.,9);

        gs_lanczos_coeff.clear();

        if(gf) clog <<"GF ";
        clog << "iteration=" << it << ":" <<endl;
        for (int q = -parameters.N; q <= parameters.N; q++)
        {
            for (int sz = -parameters.N; sz <= parameters.N; sz++)
            {
                if (qszcount[q][sz]==0) continue;
                u_n_m1[q][sz] = u_n[q][sz];
                u_n[q][sz] = u_n_p1[q][sz];
            }
        }
        Hu_n=H_dot_u( parameters, basis, qszcount, u_n );
        //calculate a_n=<u_n|H|u_n>/<u_n|u_n>
        a.push_back( u_dot_up( parameters, basis, qszcount, u_n, Hu_n) / u_dot_up( parameters, basis, qszcount, u_n, u_n) );
        b.push_back( sqrt( u_dot_up( parameters, basis, qszcount, u_n, u_n) / u_dot_up( parameters, basis, qszcount, u_n_m1, u_n_m1) ) );
        for (int q = -parameters.N; q <= parameters.N; q++)
        {
            for (int sz = -parameters.N; sz <= parameters.N; sz++)
            {
                if (qszcount[q][sz]==0) continue;
                Matrix a_u_n_q_sz = u_n[q][sz];
                Matrix b_u_nm1_q_sz = u_n_m1[q][sz];
                a_u_n_q_sz.multiply( a[it] );
                b_u_nm1_q_sz.multiply( pow(b[it-1], 2) );
                u_n_p1[q][sz] = Hu_n[q][sz] - a_u_n_q_sz - b_u_nm1_q_sz ;
            }
        }
        energies = diagonalize_tri( a, b, gs_lanczos_coeff );

        en = energies[0];
        de = abs( en - enm1 );
        clog << "g.s. energy:" << energies[0] << "  dE_gs= " << de << endl;

        // output energy
        if ( it > 4 )  energiesOfStream << it << "\t" << energies[0]<< "\t" << energies[1]<< "\t" << energies[2] << "\t" << energies[3] <<"\t"<< energies[4] << endl;

        // check convergence
        if(de < pow(10.,-14) /*&& !gf*/) break;
        else enm1 = en;
    }

    energiesOfStream.close();
    if(gf)
       info << "ground state energy:" << endl;
    else
       info << "Green's function ground state energy:" << endl;
    info << "energy = " << energies[0] << endl;    

}

States lanczos_gs_wavefn(Parameters parameters, Basis &basis, QSzCount &qszcount, vector<double> &energies, vector<double> &gs_lanczos_coeff ,ofstream &info )
{
    RanGSL rand(1234);//initial random number generator with seed
    States u_n, u_n_p1, u_n_m1;// storing temporary lanczos basis
    States gs_wavefn;// storing the ground state wavefunction 
    vector<double> a, b;// storing the diagonal elements a and off diagonal elements b for tridiagonal matrix
    vector<double> c; //storing useless gs_lanczos_coeff
    double norm;//normalization factor for wavefunction (Note the tridiagonal matrix is in normalized lanczos basis)

    clog << "iteration=0:" << endl;
    // Set up the initial random Lanczos state u0    
    int hil_count = 0;
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if ( qszcount[q][sz]==0 ) continue;
            gs_wavefn[q][sz].resize(qszcount[q][sz],1);
            u_n[q][sz].resize(qszcount[q][sz],1);
            for (int r=0; r< qszcount[q][sz]; r++) {
                u_n[q][sz].set(r,0,rand());
                //cout << u_n[q][sz].get(r,0) << endl;
                hil_count+=1;
            }
        }
    }
    assert (pow(4.,parameters.N)==hil_count);
    clog << "total size of hilbert space:" << hil_count << endl;

    norm=sqrt( u_dot_up( parameters, basis, qszcount, u_n, u_n ));
    States Hu_n=H_dot_u( parameters, basis, qszcount, u_n ); 
    //calculate a_n=<u_n|H|u_n>/<u_n|u_n>
    a.push_back( u_dot_up( parameters, basis, qszcount, u_n, Hu_n) / pow(norm,2) );
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        { 
            if (qszcount[q][sz]==0) continue;
            for ( int r=0; r< qszcount[q][sz]; r++)
            {
                double temp =gs_lanczos_coeff[0]*u_n[q][sz].get(r,0)/norm;
                gs_wavefn[q][sz].set(r,0,temp);
            }
            Matrix a_u_n_q_sz = u_n[q][sz];
            a_u_n_q_sz.multiply( a[0] );
            u_n_p1[q][sz] = Hu_n[q][sz] - a_u_n_q_sz  ;
        }
    }

    clog << "iteration=1:" << endl;
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if (qszcount[q][sz]==0) continue;
            u_n_m1[q][sz] = u_n[q][sz];
            u_n[q][sz] = u_n_p1[q][sz];
        }
    }
    norm=sqrt( u_dot_up( parameters, basis, qszcount, u_n, u_n ));
    Hu_n=H_dot_u( parameters, basis, qszcount, u_n );
    //calculate a_n=<u_n|H|u_n>/<u_n|u_n>
    a.push_back( u_dot_up( parameters, basis, qszcount, u_n, Hu_n) / pow(norm, 2) );
    b.push_back( sqrt( pow(norm,2) / u_dot_up( parameters, basis, qszcount, u_n_m1, u_n_m1) ) );
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if (qszcount[q][sz]==0) continue;
            for ( int r=0; r< qszcount[q][sz]; r++) 
            {
                double temp = gs_wavefn[q][sz].get(r,0);
                temp += gs_lanczos_coeff[1]*u_n[q][sz].get(r,0)/norm;
                gs_wavefn[q][sz].set(r,0,temp);
            }
            Matrix a_u_n_q_sz = u_n[q][sz];
            Matrix b_u_nm1_q_sz = u_n_m1[q][sz];
            a_u_n_q_sz.multiply( a[1] );
            b_u_nm1_q_sz.multiply( pow(b[0], 2) );
            
            u_n_p1[q][sz] = Hu_n[q][sz] - a_u_n_q_sz - b_u_nm1_q_sz ;
        }
    }

    energies = diagonalize_tri( a, b, c );
    clog << "g.s. wavefunction calculation. g.s. energy:" << energies[0] << endl;
    
    for (int it=2; it<parameters.itmax; it++)
    {
        double en, enm1=energies[0], de=pow(10.,9);

        gs_lanczos_coeff.clear();
        clog << "iteration=" << it << ":" <<endl;
        for (int q = -parameters.N; q <= parameters.N; q++)
        {
            for (int sz = -parameters.N; sz <= parameters.N; sz++)
            {
                if (qszcount[q][sz]==0) continue;
                u_n_m1[q][sz] = u_n[q][sz];
                u_n[q][sz] = u_n_p1[q][sz];
                
            }
        }
        norm=sqrt( u_dot_up( parameters, basis, qszcount, u_n, u_n ));
        Hu_n=H_dot_u( parameters, basis, qszcount, u_n );
        //calculate a_n=<u_n|H|u_n>/<u_n|u_n>
        a.push_back( u_dot_up( parameters, basis, qszcount, u_n, Hu_n) / pow(norm,2) );
        b.push_back( sqrt( pow(norm,2) / u_dot_up( parameters, basis, qszcount, u_n_m1, u_n_m1) ) );
        for (int q = -parameters.N; q <= parameters.N; q++)
        {
            for (int sz = -parameters.N; sz <= parameters.N; sz++)
            {
                if (qszcount[q][sz]==0) continue;
                for ( int r=0; r< qszcount[q][sz]; r++) 
                {
                    double temp = gs_wavefn[q][sz].get(r,0);
                    temp += gs_lanczos_coeff[it]*u_n[q][sz].get(r,0)/norm;
                    gs_wavefn[q][sz].set(r,0,temp);
                }
                Matrix a_u_n_q_sz = u_n[q][sz];
                Matrix b_u_nm1_q_sz = u_n_m1[q][sz];
                a_u_n_q_sz.multiply( a[it] );
                b_u_nm1_q_sz.multiply( pow(b[it-1], 2) );
                u_n_p1[q][sz] = Hu_n[q][sz] - a_u_n_q_sz - b_u_nm1_q_sz ;
            }
        }
        energies = diagonalize_tri( a, b, c );

        en = energies[0];
        de = abs( en - enm1 );
        clog << "g.s. wave function calculation. g.s. energy:" << energies[0] << "  dE_gs= " << de << endl;

        //check convergence
        if(de < pow(10.,-14)) break;
        else enm1 = en;
    }

    info << "Finish ground state wave function calculation" << endl;

    return gs_wavefn;
}

States c_i_d_u( Parameters parameters, Basis &basis, QSzCount &qszcount, int site, States &gs_wavefn )
{
    double small = 0.0000001, charge, spin;
    States c_i_d_u;
    bool found=false;
    //find the sector where the ground state locate
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if (qszcount[q][sz] == 0)
                continue;
            c_i_d_u[q][sz].resize(qszcount[q][sz],1);
            //c_i_d_u[q][sz].zero();
            for (int r=0; r < qszcount[q][sz]; r++) 
            {
                if ( gs_wavefn[q][sz].get(r,0) > small ) 
                {
                   charge = q;
                   spin = sz;
                   found = true;
                   break;
                }
            }
            if(found) break;
        }
        if(found) break;
    }
    clog << "g.s. wavefunction is in sector q=" << charge << " sz=" << spin << endl;
    //cout << "c_i_d_u site "<< site << " " << gs_wavefn[0][0].get(0,0) << endl;
    for (int q = -parameters.N; q <= parameters.N; q++) {
        for (int sz = -parameters.N; sz <= parameters.N; sz++) {
            for (int r=0; r<qszcount[q][sz]; r++)
                c_i_d_u[q][sz].set(r,0,0.00000000);
        }
    }

    for (int r=0; r < qszcount[charge+1][spin+1]; r++) 
    {        
        for (int rp=0; rp<qszcount[charge][spin]; rp++)
        {
            //clog << "r=" <<r <<" rp=" <<rp<<endl;
            //clog << basis[charge+1][spin+1][r][site] <<"  " <<basis[charge][spin][r][site]<<endl;
            bool same = true;// check if the other site are the same
            for (int m=0; m<parameters.N; m++)
            {
                if (m==site) continue;
                if ( basis[charge+1][spin+1][r][m] != basis[charge][spin][rp][m] )
                {
                    same = false;
                    break;
                }
                
            }
            if ( (same) && (basis[charge+1][spin+1][r][site] == 1) && (basis[charge][spin][rp][site] == 0) )
            {
                c_i_d_u[charge+1][spin+1].set( r,0,gs_wavefn[charge][spin].get(rp,0) );
            }
            if ( (same) && (basis[charge+1][spin+1][r][site] == 2) && (basis[charge][spin][rp][site] == -1) )
            {
                c_i_d_u[charge+1][spin+1].set( r,0,gs_wavefn[charge][spin].get(rp,0) );
            }
        }
    }

    return c_i_d_u;
}

States c_i_u( Parameters parameters, Basis &basis, QSzCount &qszcount, int site, States &gs_wavefn )
{
    double small = 0.0000001, charge, spin;
    States c_i_u;
    bool found=false;
    //find the sector where the ground state locate
    for (int q = -parameters.N; q <= parameters.N; q++)
    {
        for (int sz = -parameters.N; sz <= parameters.N; sz++)
        {
            if (qszcount[q][sz] == 0)
                continue;
            c_i_u[q][sz].resize(qszcount[q][sz],1);
            //c_i_u[q][sz].zero();
            for (int r=0; r < qszcount[q][sz]; r++) 
            {
                if ( gs_wavefn[q][sz].get(r,0) > small ) 
                {
                   charge = q;
                   spin = sz;
                   found = true;
                   break;
                }
            }
            if(found) break;
        }
        if(found) break;
    }
    clog << "g.s. wavefunction is in sector q=" << charge << " sz=" << spin << endl;
    //cout << "c_i_u site "<< site << " " << gs_wavefn[0][0].get(0,0) << endl;
    for (int q = -parameters.N; q <= parameters.N; q++) {
        for (int sz = -parameters.N; sz <= parameters.N; sz++) {
            for (int r=0; r<qszcount[q][sz]; r++)
                c_i_u[q][sz].set(r,0,0.00000000);
        }
    }

    for (int r=0; r < qszcount[charge-1][spin-1]; r++) 
    {        
        for (int rp=0; rp<qszcount[charge][spin]; rp++)
        {
            //clog << "r=" <<r <<" rp=" <<rp<<endl;
            //clog << basis[charge+1][spin+1][r][site] <<"  " <<basis[charge][spin][r][site]<<endl;
            bool same = true;// check if the other site are the same
            for (int m=0; m<parameters.N; m++)
            {
                if (m==site) continue;
                if ( basis[charge-1][spin-1][r][m] != basis[charge][spin][rp][m] )
                {
                    same = false;
                    break;
                }
                
            }
            if ( (same) && (basis[charge-1][spin-1][r][site] == 0) && (basis[charge][spin][rp][site] == 1) )
            {
                c_i_u[charge-1][spin-1].set( r,0,gs_wavefn[charge][spin].get(rp,0) );
            }
            if ( (same) && (basis[charge-1][spin-1][r][site] == -1) && (basis[charge][spin][rp][site] == 2) )
            {
                c_i_u[charge-1][spin-1].set( r,0,gs_wavefn[charge][spin].get(rp,0) );
            }
        }
    }

    return c_i_u;
}
vector<double> diagonalize_tri(vector<double> diag, vector<double> offdiag, vector<double> &gs_lanczos_coeff ) 
{

    int N = diag.size();
    //vector<double> eigenvalues(N);
    double eigenvectors[N*N];
    char v = 'V';
    int info, lwork = max(1, 3*N-1);
    double *work = NULL;

    work = new double[lwork];
    utils::dstev_(&v, &N, &diag.at(0), &offdiag.at(0), eigenvectors , &N ,work, &info);

    if (info != 0)
      cout << "diagonalization routine dsyev_ error: info = " << info << endl;Matrix c;

    for(int i=0; i<N; i++) gs_lanczos_coeff.push_back(eigenvectors[i]);//push_back ground state lanczos coefficient

    delete[] work;
    return diag;//eigenvalues;

    //cout << "The eigenvectors are:" << endl;
    //for(int i=0; i<N*N;i++) cout << eigenvectors[i] <<"\t";
    //cout<< endl;
}

vector<complex<double> > greenfunction(int e_or_h, Parameters parameters, Basis &basis, QSzCount &qszcount, vector<double> &a, vector<double> &b, double b0 , double e0, vector<double> &omega)
{
    vector<complex<double> > gf(parameters.omegaPoints);//green's function

    clog << "E0=" << e0 << endl;
    /*vector<double> ap = a;
    vector<double> bp;
    bp.push_back(b0);
    for (int i=0; i<b.size(); i++) bp.push_back(b[i]);
    clog << "size of a=" << ap.size() << "  size of b=" << bp.size() << endl;*/

    // set up the grid for the frequency values
    for (int i = 0; i < parameters.omegaPoints; i++) {
        vector<double> ap;
        for (int j=0; j<a.size(); j++) ap.push_back(a[a.size()-1-j]);
        vector<double> bp;
        complex<double> z;
        for (int j=0; j<b.size(); j++) bp.push_back(b[b.size()-1-j]);
        bp.push_back(b0);
        //clog << "size of a=" << ap.size() << "  size of b=" << bp.size() << endl;
        //cout << complex<double>( omega[i]+e0, parameters.broadening ) << endl;
        if (e_or_h == 1)
           z = complex<double>( omega[i]+e0, parameters.broadening ); 
        else
           z = complex<double>( omega[i]-e0, parameters.broadening );
        //cout << z << endl; 
        gf[i] = continuefraction( e_or_h, z , ap , bp );
        //cout << omega[i] << "\t"  << real(gf[i]) << "\t" << imag(gf[i]) << endl; 
    }

    return gf;
}

complex<double> continuefraction(int e_or_h, complex<double> &z, vector<double> &a, vector<double> &b )
{
    /*if ( a.size()==1 && b.size()==1 ) {
       double an = a.back();
       a.pop_back();
       double bn = b.back();
       b.pop_back();
       if( e_or_h == 1 )
         return pow( bn , 2 )/ ( z - an );
       else
         return pow( bn , 2 )/ ( z + an );
    }*/
    if ( a.size()==0 && b.size()==0){
       //clog <<"a size=" << a.size() <<"  b size="<< b.size() <<endl;
       return 0;
    }
    else {
       //clog <<"a size=" << a.size() <<"  b size="<< b.size() <<endl;
       complex<double> an = complex<double>(a.back() ,0 );
       a.pop_back();
       complex<double> bn = complex<double>(b.back() ,0 );
       //clog << an << "\t" <<bn<<endl;
       b.pop_back();
       if( e_or_h == 1 )
         return pow( bn , 2 )/ ( z - an - continuefraction( e_or_h , z , a, b ) );
       else
         return pow( bn , 2 )/ ( z + an - continuefraction( e_or_h , z , a, b ) );
    }
}
