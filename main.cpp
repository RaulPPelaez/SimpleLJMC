// Raul P. Pelaez 2016. 
// Molecular Dynamics Simulation of a Lennard Jonnes liquid with PBC and Neighbour head and list
// Compile with: make
// Usage: ./mc
//
// The program outputs a file with the positions of all the particles with frequency save_freq. This file is spunto (superpunto) compatible! just spunto data.pos!
//and a two column file containing the averaged radial function distribution
//
// Change the parameters in the code below (search Parameters)!
//
//Notes:
//
//We store positions in a 1D array, as x1, y1, z1, x2, y2...
//This way the k coordinate of particle i is accesible through pos[3*i+k]
//
//The positions are used from -0.5 to 0.5, we only acknowledge the box size L 
//when computing the potential interaction or writing to the disk. This simplifies some things
//
//We let the positions grow without taking into considerations PBC, maybe we want to study difusion!. We only use minimum image convection (reduce to the promary box applying PBC) when needed, that is when computing the potential or the cell the particle is in.
//
//Change it as you like! this version is intended for teaching, so it can be highly improved!
//
//You can send me questions and feedback at raul.perez@uam.es
//
//CHANGES FROM THE PREVIOUS MC CODE!
//From the previous MC assignement code you need very few changes to make a MD code!.
//
//Basically you need two new variables to store the forces acting 
// on a particle and their velocities.
//Then, you need to adapt the functions energyNL (now forcesNL) and LJNL 
// to return three values, the three components of the force acting
// on particle index (rather than the energy as before).
//Finally, change do_step to do the following:
//     1- Compute the forces acting on every particle
//     2- Use that forces to integrate the next time step using velocity verlet,
//        updating the positions and fixing head and list.
//
//That is it!!
//
//
//I changed the function names and arguments and made the new variables
// , so you only have to:
//     1- Write your code in the functions compute_forces() and 
//        update_positions(). 
//     2- Modify the LJNL code to sum the force in each coordinate (fx, fy ,fz) 
//        instead of computing and returning the potential.
//
//And again, you can contact me at raul.perez@uam.es or at the
// github page https://github.com/RaulPPelaez/SimpleLJMC if you find something odd!
//
#include<iostream>  //cout, printf
#include<vector>    //vector class
#include<fstream>   //ofstream, ifstream for file IO
#include<cmath>     //sqrt for rdf
#include<algorithm> //sort algorithm for heal_list
using namespace std;


//** Molecular Dynamics functions **//
void init();      //Initialices all variables and sets up initial conditions
void do_step();   //Computes the forces on every particle and moves them
void compute_forces(); //do_step calls these two functions
void update_positions(int sw); //Velocity verlet integrator, sw==1 is the first step and sw==2 the second 
void forceNL(int index);//Computes the force acting on particle index
void LJNL(float *r1, float *r2, float &fx, float &fy, float &fz); //Lennard Jonnes force between two positions, taking into account PBC, the result is stored summed tu fx, fy, fz
void make_linked_list();  //Fills head and list with the current system state
void heal_list(int cella, int cellb); //Recalculates cella and cellb in head and list   


//** Helper functions **//
void  apply_pbc(float *v);  //Reduces a vector position to the -0.5, 0.5 range
int   pbc_cells(int icell); //Returns icell inside the 0, ncells-1 range
int   getcell(float *pos, int *cell); //Computes cell coordinates in cell and returns cell index 
int   getcell(float *pos);     //Computes cell index for head
float norm(const float *v);    // returns |v|^2
void  rdf(bool write = false); //Compute the radial function distribution, write it if write=true
void  write_results();      //Writes all positions to disk, in spunto format
void  print_wellcome();     //Prints awellcome message


//**Monte Carlo variables**//
//We are not changing the number of particles (nor cells) in this code, so we dont even need vector<>, but it is useful! 
vector<float> pos;  //Particle positions, stored aligned as x1,y1,z1,x2,y2,z2...
vector<float> F, vel;  //Particle forces and velocities, stored aligned as x1,y1,z1,x2,y2,z2...
vector<int>   list, head; //Neighbour list head and list
int           ncells;     //Number of cells in one direction in the head and list grid (ncellstotal = ncells ^3)  
float jump_size;          //Step length when attemting to move a particle


//**Parameters**//
//You can write your code in init() to read all the parameters from a file i.e!
int   N = 100;         //Number of particles
float L = 10;          //Simulation box size (units of sigma)
float rcut = 2.25;     //Interaction cut off (units of sigma)
const float dt = 0.02; //Integration time step


const int nsteps         = 10000;  //Number of simulation steps to perform (one steps is N tries)
const int save_freq      = 100;   //Write results every save_freq steps
const int thermal_nsteps = 1;    //Number of thermalization steps

//**Radial function distribution parameters**//

const int  RDF_SAMPLES =  125;   //Number of bins
const float RDF_CUTOFF = 5.0f;   //Maximum distance to compute (try distances > L/2)
vector<float> gdr;


//***********************MAIN******************//
int main(){
  /*Initialize*/
  init();

  /*Thermalize the system by doing some steps*/
  for(int t=0; t<thermal_nsteps; t++)
    do_step();
  
  int print_steps= nsteps/100+1; //For printing progress
  /*Main loop, for the number of  steps requested*/
  for(int t=0; t<nsteps; t++){
    /*Print progress*/
    if(t%print_steps==0){
      printf("\r Simulation in progress %d%%", int(100.0f*t/(float)nsteps+0.5f));
      fflush(stdout);
    }
    /*Average radial function distribution*/
    rdf(); 
    /*Perform N MC updates*/
    do_step();
    /*Write results to disk"*/
    if(t%save_freq==0) 
      write_results();
  }
  /*Write gdr to disk*/
  rdf(true);
  printf("\r Simulation in progress 100%%\nDONE!!\n Exiting\n");
  return 0;
}

//********Initialize********//
void init(){
  gdr.resize(RDF_SAMPLES,0); /*Reserve space and initialize to zero the radial function distribution*/
  pos.resize(3*N); /*Reserve space for positions*/
  F.resize(3*N,0); /*Reserve space for forces*/
  vel.resize(3*N,0); /*Reserve space for vel*/
  
  /*Initial position*/
  //We take the box size, L, as the unit of length, simplifying some things.
  //When computing the LJ interaction, we multiply by L.
  ifstream in("kk");
  float x, y, z, trash;
  for(int i=0; i<N; i++){ //Positions go from -0.5 to 0.5
    in>>x>>y >> z>>trash>>trash;
    pos[3*i] = x/L;
    pos[3*i+1] = y/L;
    pos[3*i+2] = z/L;
    
  }
  in.close();

  /*Set up head and list variables*/
  //ncells is calculated so there can not be
  //distances greater than rcut between particles in two consecutive cells
  ncells = int(L/rcut + 0.5);
  head.clear();
  list.clear();
  //We store and additional element in head and list to be able to address them from 1 to ncellst
  // and 1 to N respectively (so head[0] and list[0] are not used), this makes the algorithm more readable later.
  head.resize(ncells*ncells*ncells+1, 0);
  list.resize(N+1, 0);
  /*Compute head and list*/
  make_linked_list();
  print_wellcome();
}

//******Clears and fills head and list**********//
//Be careful with indices in this algorithm, 0 has a special meaning in head and list,
// (no particles in that cell and last particle in cell respesctively). So we can not 
// identify the particles from 0 to n-1, but from 1 to n. We could also use 0,n-1 and 
// take -1 i.e. as no particle identifier, but the other approach is much less troublesome. 
void make_linked_list(){
  /*fill with zeros*/
  //list will be overwritten, son we dont have to fill it with 0!
  std::fill(head.begin(), head.end(), 0);

  int icell;   //Cell index in head
  float temppos[3]; //position of a particle, we dont apply PBC to pos variable
  /*For every particle in the system (carefull with the indices!)*/
  //We go from 1 to N, but address the particles as 0 to N-1.
  for(int i=1; i<=N; i++){ 
    /*Save the (i-1)th position to tempos*/
    for(int j=0; j<3; j++) temppos[j] = pos[3*(i-1)+j];
    /*And reduce it to the primary box*/
    apply_pbc(temppos);
    /*Compute the cell coordinates of particle i-1, see getcell below!*/ 
    /*Compute the head index of cell[] (Look in the notes!)*/
    icell = getcell(temppos);
    /*Add particle to head and list (Look in the notes!)*/
    list[i] = head[icell];
    head[icell] = i;
  }
}             
//**********Recalculates cella and cellb in head and list***************//
//This algorithm can probably be writeen much better, but man...
//In principle we only need to take one particle from cell a and put it in cell b.
//In this function we just obliterate both cells and relocate every particle in both.

//A FEW WORDS ABOUT PERFORMANCE
//On the other hand, I did some profiling using valgrind (without calling rdf, see prof.png)
// and saw that this function is basically free. To give some numbers:
//                         heal_list takes 0.1% of the total execution time, surprising!
//                         wheareas LJNL (the most expensive function!) takes 72%
//                The rest of the execution is basicaly the rest of energyNL.
//So even when heal list looks like in can be improved, you can see how any minor tweak to LJNL will be
// directly noted in the total execution time, wheareas improving heal_list... not so much.
//This is because LJNL is called thousands of times more than the other functions, so it is the target 
// of optimization!
void heal_list(int cella, int cellb){
  //Vector storing the particles to relocate, size N to be extra cautious!
  //static because we only need to initialize it once!
  static vector<int> redo(N, 0);
  
  int to_redo=0; //Number of particles to fix
  int icell; //Index of the cell in head
  
  /*Take the lead particle in cella*/
  int i = head[cella];
  /*Drop down though list, storing every particle in redo*/
  while(i!=0){
    redo[to_redo]=i;
    i = list[i];
    to_redo++;
  }
  /*Same with cellb*/
  i = head[cellb];
  while(i!=0){
    redo[to_redo]=i;
    i = list[i];
    to_redo++;
  }
  /*This vector needs to be sorted to do the trick!*/
  std::sort(redo.begin(), redo.begin()+to_redo);
  /*Reset both cells*/
  head[cella] = head[cellb] = 0;
  /*Now we use the same old algorithm that we use to create the head and list*/
  float temppos[3];
  for(int j=0; j<to_redo; j++){
    i = redo[j];//Only now we do not go though every particle, just the ones in redo!
    for(int k=0; k<3; k++) temppos[k] = pos[3*(i-1)+k];
    apply_pbc(temppos);
    icell=getcell(temppos);
    list[i] = head[icell];
    head[icell] = i;
  }

}


//*********Perform one MC step***********//
// **** CHANGE HERE TO ADAPT TO VELOCITY VERLET INTEGRATOR //
void do_step(){
  /*Set forces to zero*/
  std::fill(F.begin(), F.end(), 0);
  update_positions(1); //first velocity verlet step (fill)
  /*This function should compute the forces 
    acting on every particle and store that information in F */
  compute_forces();
  /*This function should use the forces (F variable) now filled to 
    integrate the positions, updating the pos and vel variables*/
  update_positions(2);  //second velocity verlet step (fill)
  
}

void compute_forces(){
  //You can use the already declared functions from the MC assignement here
  //With very little change you can adapt energyNL to update the F variable
  // (three components of F per particle!)
  //the same with LJNL, wich should now return three components, (fx, fy and fz).

  //If you do that, the code is just as simple as this.
  for(int i=0; i<N; i++)
    forceNL(i);
  // YOU CAN TRY TO IMPROVE THE FORCE EVALUATION BY looping over all linked-cells in forceNL
}
void update_positions(int sw){
  //  CHANGE HERE TO VERLET***
  if(sw==1){
    //First verlet step
  }
 else{
   //secod verlet step
 // THIS IS AN EULER INTEGRATOR (just to see it works, after delete it)
  //Put your velocity verlet code here, you need the forces, 
  //velocities and positions of the particles, plus the time step
  //You can access all that data in the F, vel, pos and dt variables.
  //You also have available all the parameters, like the number of particles, N
  for(int i = 0; i<N; i++)
    for(int j=0; j<3; j++){
      vel[3*i+j] += F[3*i+j]*dt;
      pos[3*i+j] += vel[3*i+j]*dt;
    }
  //Be carefull, the particles might have changed cells!
  make_linked_list();
  }
  }



//******Compute the energy of the system for particle index using head and list ****//
//It is easier than it seems, we have to go through particle particle index sees, but only in the
// same cell that the particle is in, plus the first neighbour ones.
void forceNL(int index){
  /*Save the index particle position*/
  float posindex[3];//Temporal position storage
  for(int i=0; i<3; i++) posindex[i] =  pos[3*index+i];
  /*Get it to the primary box*/
  apply_pbc(posindex);

  int cell[3];   //Cell coordinates of the index particle
  getcell(posindex, cell);

  int j; //Index of neighbour particle
  int jcel, jcelx, jcely, jcelz; //cell coordinates and cell index for particle j
  /*For every neighbouring cell (26 cells in 3D)*/
  for(int jx=cell[0]-1; jx<=cell[0]+1;jx++)
    for(int jy=cell[1]-1; jy<=cell[1]+1;jy++)
      for(int jz=cell[2]-1; jz<=cell[2]+1;jz++){
	/*The neighbour cell must take into account pbc! (see pbc_cells!)*/
	jcelx = pbc_cells(jx);
	jcely = pbc_cells(jy);
	jcelz = pbc_cells(jz);
	//See getcell!
	jcel =jcelx + (jcely-1)*ncells+(jcelz-1)*ncells*ncells;
	/*Get the highest index particle in cell jcel*/
	j = head[jcel]; 
	/*If there is no particles go to the next cell*/
	if(j==0) continue;
	/*Use list to travel through all the particles, j, in cell jcel*/
	do{
	  /*Add the energy of the pair interaction*/
	  //Be careful not to compute one particle with itself!, j-1 because of head and list indexes!
	  if(index!=(j-1))
	    LJNL(&pos[3*index], &pos[3*(j-1)], F[3*index], F[3*index+1], F[3*index+2]);
	  j=list[j];
	  /*When j=0 (list[j] = 0) then there is no more particles in cell jcel (see the notes!)*/
	}while(j!=0);
      } 
}





//**Give me the cell index in head taking into account PBC**//
int pbc_cells(int icell){
  if(icell==0)              return ncells;
  else if(icell==ncells+1)  return 1;
  else                      return icell;
}


//**Lennard Jonnes interaction between two points**//
//This is  where the program spends most time in (~70% of the time!), it is called many many times.
//The most expensive thing is rij = r2-r1 and applying PBC.
void LJNL(float *r1, float *r2, float &fx, float &fy, float &fz){
  /*Take a vector joining twe two points*/
  float rij[3];
  for(int i=0; i<3; i++) rij[i] = r2[i]-r1[i];
  /*Reduce it two te minimum image convection*/
  apply_pbc(rij);
  /*And compute the LJ potential using its norm!*/
  float r2mod = norm(rij)*L*L; //Now the distance is between 0 and L/2 ! 
  if(r2mod>rcut*rcut) return;
  double invr6 = pow(1.0f/r2mod, 3);
  /*Here sigma = 1 and epsilon = 1*/
  //Check this out, you dont even need to compute sqrt(r2)!
  float fmoddivr = -48.0*invr6*(invr6-0.5)/r2mod; //F/r

  //F*rij/r being F and rij vectors!
  fx += fmoddivr*rij[0]; 
  fy += fmoddivr*rij[1];
  fz += fmoddivr*rij[2];
}


//*******Apply PBC to a position (magic inside)*******//
void apply_pbc(float *r){
  //This algorithm is a little bit tricky, look in the notes!
  //We make use of the fact that positions go from -0,5 to 0.5 and integer arithmetics
  //Imagine this:
  /*    
   *    __<-Simulation box 
   *   |  |  |. <-A particle between 1.5 and 2.0 (two boxes to the right) 
   *   |__|__|__  
   *-0.5 0.5
   *
   *  In integer arithmetics that int([1.5, 2.0]+0.5) = 2
   *  And going back to float arithmetics [1.5, 2.0]-2 = [-0.5, 0.5] Yay!!
   *
   *  This is what I am doing here V. Only the logic is with -0.5 when r<-0.5
   */
  for(int i=0; i<3; i++)
    r[i]-=int( ( (r[i]<0)?-0.5:0.5 ) + r[i]);
}

//**Computes cell coordinates and cell index**//
int getcell(float *pos, int *cell){
  int icell; //Cell index for head
  //We make use of the fact that pos is between -0.5 and 0.5.
  //Thanks to integer arithmetics (C++ truncates to transform float to int)
  //cell[i] is in the range 1,2,...ncells
  for(int j=0; j<3; j++) cell[j] = 1 + (0.5 + pos[j])*ncells;  
  //This is a little bit tricky, we have to transform a 3D coordinate (cell) to a 1D one (icell)
  //It is the same that with the position storage. See the notes!
  icell=cell[0]+(cell[1]-1)*ncells+(cell[2]-1)*ncells*ncells;
  return icell;
}
int getcell(float *pos){ //Maybe you dont need cell coordinates
  int icell; //Cell index for head
  int cell[3]; //mx, my and mz of a particle
  for(int j=0; j<3; j++) cell[j] = 1 + (0.5 + pos[j])*ncells;  
  icell=cell[0]+(cell[1]-1)*ncells+(cell[2]-1)*ncells*ncells;
  return icell;
}



//**Return the norm of a vector**//
float norm(const float *v){
  float mod;
  for(int i=0; i<3; i++) mod += v[i]*v[i];
  return mod;
}


//**Computes the radial function distribution**//
//Caution!, This is a very expensive function!. It is O(N^2)
void rdf(bool write){
  static int avg_counter = 1;                  //counter to average
  float rijmod;                                // distance between two particles
  float ratio = RDF_CUTOFF/(float)RDF_SAMPLES; //dr, distance interval of one bin
  float rij[3];                                //rj-ri

  int bin=0; //Bin in wich a pair is
  /*For all particle pairs*/
  for(int i=0; i<N-1; i++) 
    for(int j=i+1; j<N; j++){
      /*Compute the distance*/
      for(int k=0; k<3; k++) rij[k] = pos[3*j+k]-pos[3*i+k];
      /*Reduce it two te minimum image convection*/
      apply_pbc(rij);
      rijmod = sqrt(norm(rij))*L; //Now the distance is between 0 and L/2 ! 
      /*If they are in range compute the bin and sum to gdr*/
	if(rijmod<RDF_CUTOFF){
	  //Integer arithmetics magic again! in this type of algorithm
	  // the +0.5 makes C++ to round to the highest (so >=0.5 (float) goes to 1 (int) and <0.5 to 0)
	  //We are basically asking, how many times does ratio fits in rijmod?, and thats directly the bin index!
	  bin=int((rijmod/ratio) +0.5);
	  gdr[bin]++;
	}	   
    }
  /*Acknowledge that this function has been called one more time for averaging*/
  avg_counter++;
  /*If you want to write the results:*/
  if(write){
    /*You have to normalize gdr, see the notes!*/
    //You will see how g(r)->1 when r is large (and 0 if r> L/2)
    float normalization = avg_counter*2.0f*M_PI*ratio*N*N/(L*L*L);
    float r;
    ofstream gout("gdr.dat");
    for(int i=1; i<RDF_SAMPLES; i++){
      r = i*ratio;
      gout<<r<<" "<<(float)gdr[i]/(normalization*r*r)<<"\n";
    }
     gout.close();
  }
}



/*Each time this function is called,
  the current positions of all the particles are attached to data.pos*/
void write_results(){
  static ofstream out("data.pos"); //Static so each step is appended to the file
  out<<"#\n"; //Superpunto syntax
  float temppos[3];
  for(int i=0; i<N; i++){
    for(int j=0; j<3; j++) temppos[j] = pos[3*i+j];
    apply_pbc(temppos);
    //4th column is the radius, 2^(1/6) = 0.56123*2 is the distance at the minimum of energy in LJ 
    out<<temppos[0]*L<<" "<<temppos[1]*L<<" "<<temppos[2]*L<<" 0.56123 1\n";
  }
  out.flush();
}


/*Prints some information about the simulation being performed*/
void print_wellcome(){
  printf("Lennard-Jonnes liquid Molecular Dynamics simulation!\n\n");

  printf("Simulation parameters:\n");
  printf("\tNumber of particles: %d\n\tSimulation box size: %.2f sigma\n\n",
	                   N,                         L);
  printf("\tTotal number of cells: %d\n", ncells*ncells*ncells);
  printf("\n");
}

