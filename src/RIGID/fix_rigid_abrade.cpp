// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "fix_rigid_abrade.h"

#include "atom.h"
#include "atom_vec.h"
#include "atom_vec_body.h"
#include "atom_vec_ellipsoid.h"
#include "atom_vec_line.h"
#include "atom_vec_tri.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"

#include "fix_wall_gran.h"
#include "fix_wall_gran_region.h"
#include "granular_model.h"

#include "group.h"

#include "input.h"
#include "math_const.h"
#include "math_eigen.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "molecule.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "output.h"
#include "random_mars.h"
#include "random_park.h"
#include "region.h"
#include "respa.h"
#include "rigid_const.h"
#include "tokenizer.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>
#include <unordered_map>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string>    // for string class
#include <utility>
#include <vector>
#include <algorithm> 

using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;
using namespace RigidConst;

static constexpr int RVOUS = 1;   // 0 for irregular, 1 for all2all

/* ---------------------------------------------------------------------- */

FixRigidAbrade::FixRigidAbrade(LAMMPS *lmp, int narg, char **arg) :
    Fix(lmp, narg, arg), step_respa(nullptr), inpfile(nullptr), body(nullptr), bodyown(nullptr),
    bodytag(nullptr), atom2body(nullptr), vertexdata(nullptr), list(nullptr), xcmimage(nullptr),
    displace(nullptr), unwrap(nullptr), eflags(nullptr),
    avec_ellipsoid(nullptr), avec_line(nullptr), avec_tri(nullptr), counts(nullptr),
    itensor(nullptr), mass_body(nullptr), langextra(nullptr), random(nullptr), id_dilate(nullptr),
    id_gravity(nullptr), onemols(nullptr)
{
  int i;

  if (comm->me == 0) utils::logmesg(lmp,"Fix {} setup ...\n", style);

  scalar_flag = 1;
  extscalar = 0;
  global_freq = 1;
  time_integrate = 1;
  rigid_flag = 1;
  virial_global_flag = virial_peratom_flag = 1;
  thermo_virial = 1;
  create_attribute = 1;
  dof_flag = 1;

  stores_ids = 1;
  dynamic_flag = 1;
  centroidstressflag = CENTROID_AVAIL;
  
  remesh_flag = 0;
  initial_remesh_flag = 0;
  check_threshold_flag = 0;

  offset_flag = 0;
  global_normals_flag = 0;

  restart_peratom = 1;    //~ Per-atom information is saved to the restart file
  peratom_flag = 1;
  size_peratom_cols = 8;    //~ normal x/y/z, area and displacement speed x/y/z, cumulative displacement
  peratom_freq = 1;    // every step, **TODO change to user input utils::inumeric(FLERR,arg[5],false,lmp);
  create_attribute = 1;    //fix stores attributes that need setting when a new atom is created

  MPI_Comm_rank(world, &me);
  MPI_Comm_size(world, &nprocs);

  // perform initial allocation of atom-based arrays
  // register with Atom class

  extended = 0;
  bodyown = nullptr;
  bodytag = nullptr;
  atom2body = nullptr;
  vertexdata = nullptr;
  xcmimage = nullptr;
  displace = nullptr;
  unwrap = nullptr;
  eflags = nullptr;
  FixRigidAbrade::grow_arrays(atom->nmax);
  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);

  // parse args for rigid body specification

  int *mask = atom->mask;
  tagint *bodyID = nullptr;
  int nlocal = atom->nlocal;

  if (narg < 7) error->all(FLERR, "1 Illegal fix rigid/abrade command");

  hstr = mustr = densitystr = nullptr;
  hstyle = mustyle = densitystyle = CONSTANT;

  if (utils::strmatch(arg[3], "^v_")) {
    hstr = utils::strdup(arg[3] + 2);
    hstyle = EQUAL;
  } else {
    hardness = utils::numeric(FLERR, arg[3], false, lmp);
    // Conver from pressure units to force/distance^2
    hardness /= force->nktv2p;
    hstyle = CONSTANT;
  }
  if (utils::strmatch(arg[4], "^v_")) {
    mustr = utils::strdup(arg[4] + 2);
    mustyle = EQUAL;
  } else {
    hardness_ratio = utils::numeric(FLERR, arg[4], false, lmp);
    mustyle = CONSTANT;
  }
  if (utils::strmatch(arg[5], "^v_")) {
    densitystr = utils::strdup(arg[5] + 2);
    densitystyle = EQUAL;
  } else {
    density = utils::numeric(FLERR, arg[5], false, lmp);
    // Convert units
    density /= force->mv2d;
    densitystyle = CONSTANT;
  }
  if (strcmp(arg[6], "molecule") == 0) {
    if (atom->molecule_flag == 0)
      error->all(FLERR, "Fix rigid/abrade requires atom attribute molecule");
    bodyID = atom->molecule;

  } else if (strcmp(arg[6], "custom") == 0) {
    if (narg < 8) error->all(FLERR, "2 Illegal fix rigid/abrade command");
    bodyID = new tagint[nlocal];
    customflag = 1;

    error->all(FLERR, "Fix rigid/abrade does not support the custom command, only the molecule body style.");

    } else
    error->all(FLERR, "3 Illegal fix rigid/abrade command");

  if (atom->map_style == Atom::MAP_NONE)
    error->all(FLERR, "Fix rigid/abrade requires an atom map, see atom_modify");

  // maxmol = largest bodyID #

  maxmol = -1;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) maxmol = MAX(maxmol, bodyID[i]);

  tagint itmp;
  MPI_Allreduce(&maxmol, &itmp, 1, MPI_LMP_TAGINT, MPI_MAX, world);
  maxmol = itmp;

  // number of linear molecules is counted later
  nlinear = 0;

  // parse optional args

  int seed;
  langflag = 0;
  inpfile = nullptr;
  onemols = nullptr;
  reinitflag = 1;

  tstat_flag = 0;
  pstat_flag = 0;
  allremap = 1;
  id_dilate = nullptr;
  t_chain = 10;
  t_iter = 1;
  t_order = 3;
  p_chain = 10;

  pcouple = NONE;
  pstyle = ANISO;

  for (i = 0; i < 3; i++) {
    p_start[i] = p_stop[i] = p_period[i] = 0.0;
    p_flag[i] = 0;
  }

  int iarg = 7;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "langevin") == 0) {
      if (iarg + 5 > narg) error->all(FLERR, "4 Illegal fix rigid/abrade command");
      if ((strcmp(style, "rigid/abrade") != 0) && (strcmp(style, "rigid/nve/abrade") != 0) &&
          (strcmp(style, "rigid/nph/abrade") != 0))
        error->all(FLERR, "5 Illegal fix rigid/abrade command");
      langflag = 1;
      t_start = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      t_stop = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      t_period = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      seed = utils::inumeric(FLERR, arg[iarg + 4], false, lmp);
      if (t_period <= 0.0) error->all(FLERR, "Fix rigid/abrade langevin period must be > 0.0");
      if (seed <= 0) error->all(FLERR, "6 Illegal fix rigid/abrade command");
      iarg += 5;

    } else if (strcmp(arg[iarg], "infile") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "7 Illegal fix rigid/abrade command");
      delete[] inpfile;
      inpfile = utils::strdup(arg[iarg + 1]);
      restart_file = 1;
      reinitflag = 0;
      iarg += 2;

    } else if (strcmp(arg[iarg], "reinit") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      reinitflag = utils::logical(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg], "mol") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      int imol = atom->find_molecule(arg[iarg + 1]);
      if (imol == -1) error->all(FLERR, "Molecule template ID for fix rigid/abrade does not exist");
      onemols = &atom->molecules[imol];
      nmol = onemols[0]->nset;
      restart_file = 1;
      iarg += 2;

    } else if (strcmp(arg[iarg], "temp") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!utils::strmatch(style, "^rigid/n.t/abrade"))
        error->all(FLERR, "Illegal fix rigid command");
      tstat_flag = 1;
      t_start = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      t_stop = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      t_period = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      iarg += 4;

    } else if (strcmp(arg[iarg], "iso") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!utils::strmatch(style, "^rigid/np./abrade"))
        error->all(FLERR, "Illegal fix rigid/abrade command");
      pcouple = XYZ;
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      p_stop[0] = p_stop[1] = p_stop[2] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      p_period[0] = p_period[1] = p_period[2] = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (domain->dimension == 2) {
        p_start[2] = p_stop[2] = p_period[2] = 0.0;
        p_flag[2] = 0;
      }
      iarg += 4;

    } else if (strcmp(arg[iarg], "aniso") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!utils::strmatch(style, "^rigid/np./abrade"))
        error->all(FLERR, "Illegal fix rigid/abrade command");
      p_start[0] = p_start[1] = p_start[2] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      p_stop[0] = p_stop[1] = p_stop[2] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      p_period[0] = p_period[1] = p_period[2] = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      p_flag[0] = p_flag[1] = p_flag[2] = 1;
      if (domain->dimension == 2) {
        p_start[2] = p_stop[2] = p_period[2] = 0.0;
        p_flag[2] = 0;
      }
      iarg += 4;

    } else if (strcmp(arg[iarg], "x") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!utils::strmatch(style, "^rigid/np./abrade"))
        error->all(FLERR, "Illegal fix rigid/abrade command");
      p_start[0] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      p_stop[0] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      p_period[0] = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      p_flag[0] = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg], "y") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!utils::strmatch(style, "^rigid/np./abrade"))
        error->all(FLERR, "Illegal fix rigid/abrade command");
      p_start[1] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      p_stop[1] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      p_period[1] = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      p_flag[1] = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg], "z") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!utils::strmatch(style, "^rigid/np./abrade"))
        error->all(FLERR, "Illegal fix rigid/abrade command");
      p_start[2] = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      p_stop[2] = utils::numeric(FLERR, arg[iarg + 2], false, lmp);
      p_period[2] = utils::numeric(FLERR, arg[iarg + 3], false, lmp);
      p_flag[2] = 1;
      iarg += 4;

    } else if (strcmp(arg[iarg], "couple") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (strcmp(arg[iarg + 1], "xyz") == 0)
        pcouple = XYZ;
      else if (strcmp(arg[iarg + 1], "xy") == 0)
        pcouple = XY;
      else if (strcmp(arg[iarg + 1], "yz") == 0)
        pcouple = YZ;
      else if (strcmp(arg[iarg + 1], "xz") == 0)
        pcouple = XZ;
      else if (strcmp(arg[iarg + 1], "none") == 0)
        pcouple = NONE;
      else
        error->all(FLERR, "Illegal fix rigid/abrade command");
      iarg += 2;

    } else if (strcmp(arg[iarg], "dilate") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix rigid/abrade nvt/npt/nph command");
      if (strcmp(arg[iarg + 1], "all") == 0)
        allremap = 1;
      else {
        allremap = 0;
        delete[] id_dilate;
        id_dilate = utils::strdup(arg[iarg + 1]);
        int idilate = group->find(id_dilate);
        if (idilate == -1)
          error->all(FLERR,
                     "Fix rigid/abrade nvt/npt/nph dilate group ID "
                     "does not exist");
      }
      iarg += 2;

    } else if (strcmp(arg[iarg], "tparam") == 0) {
      if (iarg + 4 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!utils::strmatch(style, "^rigid/n.t/abrade"))
        error->all(FLERR, "Illegal fix rigid/abrade command");
      t_chain = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      t_iter = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
      t_order = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
      iarg += 4;

    } else if (strcmp(arg[iarg], "pchain") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!utils::strmatch(style, "^rigid/np./abrade"))
        error->all(FLERR, "Illegal fix rigid/abrade command");
      p_chain = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;

    } else if (strcmp(arg[iarg], "gravity") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      delete[] id_gravity;
      id_gravity = utils::strdup(arg[iarg + 1]);
      iarg += 2;

    } else if (strcmp(arg[iarg], "stationary") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      dynamic_flag = 0;
      iarg += 1;
    } else if (strcmp(arg[iarg], "offset") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      offset_flag = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg], "normals") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      global_normals_flag = 1;
      iarg += 1;
    } else if (strcmp(arg[iarg], "remesh") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      remesh_flag = 1;
      remesh_threshold_multiplier_atom = utils::numeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else if (strcmp(arg[iarg], "equalise") == 0) {
      if (iarg + 1 > narg) error->all(FLERR, "Illegal fix rigid/abrade command");
      if (!remesh_flag)
        error->all(FLERR,
                   "Illegal fix rigid/abrade command - remesh keyword must be declared before the "
                   "equalise keyword");
      initial_remesh_flag = 1;
      
      if (update->ntimestep > 0) {
        if (me == 0) error->warning(FLERR, "fix rigid/abrade cannot equalise surface from restart file. Bodies will not be equalised.");
        initial_remesh_flag = 0;
      }

      iarg += 1;
    } else
      error->all(FLERR, "Illegal fix rigid/abrade command");
  }

  // error check and further setup for Molecule template

  if (onemols) {
    for (i = 0; i < nmol; i++) {
      if (onemols[i]->xflag == 0)
        error->all(FLERR, "Fix rigid/abrade molecule must have coordinates");
      if (onemols[i]->typeflag == 0)
        error->all(FLERR, "Fix rigid/abrade molecule must have atom types");

      // fix rigid/abrade uses center, masstotal, COM, inertia of molecule
      onemols[i]->compute_center();
      onemols[i]->compute_mass();
      onemols[i]->compute_com();
      onemols[i]->compute_inertia();
    }
  }

  // expand the vertexdata array to account for the optional normals keyword

  if(global_normals_flag) { 
      size_peratom_cols += 3;  // + optional global normals
      memory->grow(vertexdata, atom->nmax, size_peratom_cols, "rigid/abrade:vertexdata");
      array_atom = vertexdata;
    }

  // Set initial vertexdata values to zero

  for (int i = 0; i < (atom->nlocal + atom->nghost); i++) {
    vertexdata[i][0] = vertexdata[i][1] = vertexdata[i][2] = 0.0;    // normals (x,y,z)
    vertexdata[i][3] = 0.0;                                          // associated area
    vertexdata[i][4] = vertexdata[i][5] = vertexdata[i][6] = 0.0;    // displacement velocity (x,y,z) (only accessed by locally stored atoms)
    vertexdata[i][7] = 0.0;    // Cumulative displacement (only acessed by locally stored atoms)
    
    if (global_normals_flag){
          vertexdata[i][8] = vertexdata[i][9] = vertexdata[i][10] = 0.0;   // normals stored in global coordinates
    
    }
  }

  // set pstat_flag

  pstat_flag = 0;
  for (i = 0; i < 3; i++)
    if (p_flag[i]) pstat_flag = 1;

  if (pcouple == XYZ || (domain->dimension == 2 && pcouple == XY))
    pstyle = ISO;
  else
    pstyle = ANISO;

  // create rigid bodies based on molecule or custom ID
  // sets bodytag for owned atoms
  // body attributes are computed later by setup_bodies()

  double time1 = platform::walltime();

  create_bodies(bodyID);

  if (comm->me == 0)
    utils::logmesg(lmp, "  create bodies CPU = {:.3f} seconds\n", platform::walltime() - time1);

  // set nlocal_body and allocate bodies I own

  tagint *tag = atom->tag;

  nlocal_body = nghost_body = 0;
  for (i = 0; i < nlocal; i++)
    if (bodytag[i] == tag[i]) nlocal_body++;

  nmax_body = 0;
  while (nmax_body < nlocal_body) nmax_body += DELTA_BODY;
  body = (Body *) memory->smalloc(nmax_body * sizeof(Body), "rigid/abrade:body");

  // set bodyown for owned atoms

  nlocal_body = 0;
  for (i = 0; i < nlocal; i++)
    if (bodytag[i] == tag[i]) {
      body[nlocal_body].ilocal = i;
      bodyown[i] = nlocal_body++;
    } else
      bodyown[i] = -1;

  // bodysize = sizeof(Body) in doubles

  bodysize = sizeof(Body) / sizeof(double);
  if (bodysize * sizeof(double) != sizeof(Body)) bodysize++;

  // set max comm sizes needed by this fix

  comm_forward = 1 + bodysize;
  comm_reverse = 6;

  // atom style pointers to particles that store extra info

  avec_ellipsoid = dynamic_cast<AtomVecEllipsoid *>(atom->style_match("ellipsoid"));
  avec_line = dynamic_cast<AtomVecLine *>(atom->style_match("line"));
  avec_tri = dynamic_cast<AtomVecTri *>(atom->style_match("tri"));

  // compute per body forces and torques inside final_integrate() by default

  earlyflag = 0;

  // print statistics

  int one = 0;
  bigint atomone = 0;
  for (i = 0; i < nlocal; i++) {
    if (bodyown[i] >= 0) one++;
    if (bodytag[i] > 0) atomone++;
  }
  MPI_Allreduce(&one, &nbody, 1, MPI_INT, MPI_SUM, world);
  bigint atomall;
  MPI_Allreduce(&atomone, &atomall, 1, MPI_LMP_BIGINT, MPI_SUM, world);

  if (me == 0) {
    utils::logmesg(lmp,
                   "  {} rigid bodies with {} atoms\n"
                   "  {:.8} = max distance from body owner to body atom\n",
                   nbody, atomall, maxextent);
  }

  // initialize Marsaglia RNG with processor-unique seed

  maxlang = 0;
  langextra = nullptr;
  random = nullptr;
  if (langflag) random = new RanMars(lmp, seed + comm->me);

  // mass vector for granular pair styles

  mass_body = nullptr;
  nmax_mass = 0;

  // wait to setup bodies until comm stencils are defined

  setupflag = 0;

  varflag = CONSTANT;
  if (hstyle != CONSTANT || mustyle != CONSTANT || densitystyle != CONSTANT) varflag = EQUAL;
}

/* ---------------------------------------------------------------------- */

FixRigidAbrade::~FixRigidAbrade()
{
  // unregister callbacks to this fix from Atom class

  if (modify->get_fix_by_id(id)) atom->delete_callback(id, Atom::GROW);

  // delete locally stored arrays

  memory->sfree(body);

  memory->destroy(bodyown);
  memory->destroy(bodytag);
  memory->destroy(atom2body);
  memory->destroy(vertexdata);
  memory->destroy(xcmimage);
  memory->destroy(displace);
  memory->destroy(unwrap);
  memory->destroy(eflags);

  delete random;
  delete[] inpfile;
  delete[] id_dilate;
  delete[] id_gravity;

  memory->destroy(langextra);
  memory->destroy(mass_body);
}

/* ---------------------------------------------------------------------- */

int FixRigidAbrade::setmask()
{
  int mask = 0;
  mask |= INITIAL_INTEGRATE;
  mask |= FINAL_INTEGRATE;
  mask |= END_OF_STEP;
  mask |= POST_FORCE;
  // if (langflag) mask |= POST_FORCE;
  mask |= PRE_NEIGHBOR;
  mask |= POST_NEIGHBOR;
  mask |= INITIAL_INTEGRATE_RESPA;
  mask |= FINAL_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::init()
{
  triclinic = domain->triclinic;

  // need a full perpetual neighbor list for abrasion calculations
  neighbor->add_request(this, NeighConst::REQ_FULL);

  // warn if more than one rigid fix
  // if earlyflag, warn if any post-force fixes come after a rigid fix

  int count = 0;
  for (const auto &ifix : modify->get_fix_list())
    if (ifix->rigid_flag) count++;
  if (count > 1 && comm->me == 0)
    error->warning(FLERR, "More than one fix rigid command");

  if (earlyflag) {
    bool rflag = false;
    for (const auto &ifix : modify->get_fix_list()) {
      if (ifix->rigid_flag) rflag = true;
      if ((comm->me == 0) && rflag && (ifix->setmask() & POST_FORCE) && !ifix->rigid_flag)
        error->warning(FLERR, "Fix {} with ID {} alters forces after fix rigid/abrade", ifix->style,
                       ifix->id);
    }
  }

  // check variables

  if (hstr) {
    hvar = input->variable->find(hstr);
    if (hvar < 0) error->all(FLERR, "Variable name for fix rigid/abrade does not exist");
    if (!input->variable->equalstyle(hvar))
      error->all(FLERR, "Variable for fix rigid/abrade is invalid style");
  }
  if (mustr) {
    muvar = input->variable->find(mustr);
    if (muvar < 0) error->all(FLERR, "Variable name for fix rigid/abrade does not exist");
    if (!input->variable->equalstyle(muvar))
      error->all(FLERR, "Variable for fix rigid/abrade is invalid style");
  }
  if (densitystr) {
    densityvar = input->variable->find(densitystr);
    if (densityvar < 0) error->all(FLERR, "Variable name for fix rigid/abrade does not exist");
    if (!input->variable->equalstyle(densityvar))
      error->all(FLERR, "Variable for fix rigid/abrade is invalid style");
  }

  // warn if body properties are read from inpfile or a mol template file
  //   and the gravity keyword is not set and a gravity fix exists
  // this could mean body particles are overlapped
  //   and gravity is not applied correctly

  if (!id_gravity) {
    if (modify->get_fix_by_style("^gravity").size() > 0)
      if (comm->me == 0)
        error->warning(FLERR,
                       "Gravity will not be correctly applied to rigid "
                       "bodies unless the gravity keyword is used");
  }

  // error if a fix changing the box comes before rigid fix

  bool boxflag = false;
  for (const auto &ifix : modify->get_fix_list()) {
    if (boxflag && utils::strmatch(ifix->style, "^rigid"))
      error->all(FLERR, "Rigid fixes must come before any box changing fix");
    if (ifix->box_change) boxflag = true;
  }

  // add gravity forces based on gravity vector from fix

  if (id_gravity) {
    auto *ifix = modify->get_fix_by_id(id_gravity);
    if (!ifix) error->all(FLERR, "Fix rigid/abrade cannot find fix gravity ID {}", id_gravity);
    if (!utils::strmatch(ifix->style, "^gravity"))
      error->all(FLERR, "Fix rigid/abrade gravity fix ID {} is not a gravity fix style",
                 id_gravity);
    int tmp;
    gvec = (double *) ifix->extract("gvec", tmp);
  }

  // checking to see if Newton on is specified such that angles are not double counted. 
  int newton_bond = force->newton_bond;
  if (!newton_bond) {
    error->all(FLERR, "Fix rigid/abrade requires newton on to be specified in order for particle topology to be properly considered.");
  }
  // timestep info

  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dtq = 0.5 * update->dt;

  if (utils::strmatch(update->integrate_style, "^respa"))
    step_respa = (dynamic_cast<Respa *>(update->integrate))->step;

  lastcheck = -1;

  // Check that fix/wall/rigid, if present, is defined before fix/rigid/abrade in the input script
  // If the walls are defined afterwards their force contributions on atoms are not considered in FixRigidAbrade::post_force()

  int rigid_abrade_defined = 0;
  for (auto &ifix : modify->get_fix_list()) {
    if (strcmp(ifix->style, "rigid/abrade") == 0) rigid_abrade_defined = 1;
    if ((strcmp(ifix->style, "wall/gran/region") == 0) && rigid_abrade_defined)
      error->all(FLERR,
                 "Fix wall/gran/region defined after Fix rigid/abrade. Wall abrasion will not be "
                 "captured.");
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::init_list(int /*id*/, NeighList *ptr)
{
  list = ptr;
}

/* ----------------------------------------------------------------------
   setup static/dynamic properties of rigid bodies, using current atom info.
   if reinitflag is not set, do the initialization only once, b/c properties
   may not be re-computable especially if overlapping particles or bodies
   are inserted from mol template.
     do not do dynamic init if read body properties from inpfile. this
   is b/c the inpfile defines the static and dynamic properties and may not
   be computable if contain overlapping particles setup_bodies_static()
   reads inpfile itself.
     cannot do this until now, b/c requires comm->setup() to have setup stencil
   invoke pre_neighbor() to ensure body xcmimage flags are reset
     needed if Verlet::setup::pbc() has remapped/migrated atoms for 2nd run
     setup_bodies_static() invokes pre_neighbor itself
------------------------------------------------------------------------- */

void FixRigidAbrade::setup_pre_neighbor()
{

  if (reinitflag || !setupflag) {

    if (comm->me == 0 && neighbor->style != Neighbor::MULTI)
      error->one(FLERR, "fix rigid/abrade requires hybrid neighbor lists to be enabled through the multi neighbor style.");

    // if starting from a restart file then atoms need to be displaced 
    // outwards from their COM to allign with the icosohedra vertices
    if (update->ntimestep > 0) 
      pre_setup_bodies_static();

    // building topology for use in setup_bodies_static()
    neighbor->build_topology();

    // calcualting initial mass, volume, and inertia of particles from their topology
    setup_bodies_static();
    
    areas_and_normals();

    // allocating a temporary array for use in equalise_surface()
    if (remesh_flag) {
      
      // equalise_surface_array[i][average_surface_area, sum_area_minus_average_sq, variance_normalised, old variance]
      memory->create(equalise_surface_array, nlocal_body + nghost_body, 4,
                     "rigid/abrade:equalise_surface_array");
      
      //  initialising the old variance to an arbitrarily large value which will be overwritten with the first calcualted variance
      for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++)
        equalise_surface_array[ibody][3] = BIG;
    }
  }

  else
    pre_neighbor();
  
  // Recursivelly equalising surface
  if (initial_remesh_flag) equalise_surface();

  if ((reinitflag || !setupflag)) {
    if (remesh_flag) memory->destroy(equalise_surface_array);

    // store the global minimum assocaited surface area to be used as a remeshing threshold moving forward
    if (remesh_flag){
      double local_min_area_atom = BIG;
      for (int ibody = 0; ibody < nlocal_body; ibody++) {
        body[ibody].abraded_volume = 0.0;
        if (body[ibody].min_area_atom < local_min_area_atom) {
          local_min_area_atom = body[ibody].min_area_atom;
        }
      }

      MPI_Allreduce(&local_min_area_atom, &remesh_area_threshold_atom, 1, MPI_DOUBLE_INT, MPI_MINLOC, world);
     
      // printing remeshing criteria to the user
      if (me == 0){
        utils::logmesg(lmp, "fix rigid/abrade: Minimum associated area = {:.3f}, threshold multiplier = {:.3f}, remesh threshold = {:.3f}\n", remesh_area_threshold_atom, remesh_threshold_multiplier_atom, (remesh_area_threshold_atom * remesh_threshold_multiplier_atom));
      }
        check_threshold_flag = 1;
    }


      // optionally storing normals in global coords for visualisation
      if (global_normals_flag){
        double bodynormals[3], global_normals[3];  
        for (int i = 0; i < atom->nlocal; i++) {
        
          // Checking that atom i is in a rigid body
        if (atom2body[i] < 0) continue;

        bodynormals[0] = vertexdata[i][0];
        bodynormals[1] = vertexdata[i][1];
        bodynormals[2] = vertexdata[i][2];

        Body *b = &body[atom2body[i]];
        MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, bodynormals, global_normals);

        vertexdata[i][8]  = global_normals[0];
        vertexdata[i][9]  = global_normals[1];
        vertexdata[i][10] = global_normals[2];
      }
    }

    // Setting up rigid body dynamics with respect to the new topology (maybe overwritten by readfile for bodies in the .rigid restart file)
      setup_bodies_dynamic();

    // offsetting atoms towards the COM such that the tetrahedra vertices lay on the surface of the particle
    for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) body[ibody].body_offset_flag = 1;
    offset_vertices_inwards();
    for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) body[ibody].body_offset_flag = 0;

    // If starting from a restart file neighbor information must be reset for atoms that
    // have been pushed outside of the simulatoion cell prior to building lists  is required 
    // to ensure stability when using restart files with bodies straddling periodic boundaries

    // Additionally after offsetting atoms inwards, neighbor information must also be reset for similar reasons. 
        
    // Getting a list of fixes so their pre_neighbor() and post_neighbor() functions can also be called
    std::vector<Fix *> fix_list = modify->get_fix_list();
    // rebuilding neighbor lists
    for (auto &fix_i : fix_list) fix_i->pre_exchange();

    if (domain->triclinic) domain->x2lamda(atom->nlocal+atom->nghost);
    
    domain->pbc();
    domain->reset_box();
    comm->setup();
    neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);

    for (auto &fix_i : fix_list) fix_i->pre_neighbor();
    neighbor->build(1);
    for (auto &fix_i : fix_list) fix_i->post_neighbor();

    neighbor->ago = 0;
    
    // Optionally read in a fix specific restart file which preserves some per-atom and per-body between runs
    if (inpfile) 
      readfile();
    
    // printing body vcm and acm
    for (int ibody = 0; ibody < 1; ibody++) {
      utils::logmesg(lmp, "fix rigid/abrade: Initial rigid body properties for body id {}, inertia: ({:.4f}, {:.4f}, {:.4f}), volume: {:.4f}, density: {:.4f}, mass: {:.4f}, xcm: ({:.4f}, {:.4f}, {:.4f})\n", atom->tag[body[ibody].ilocal], body[ibody].inertia[0], body[ibody].inertia[1], body[ibody].inertia[2], body[ibody].volume, body[ibody].density, body[ibody].mass, body[ibody].xcm[0], body[ibody].xcm[1], body[ibody].xcm[2]);
      utils::logmesg(lmp, "fix rigid/abrade: Initial dynamics for body id {}, vcm: ({:.4f}, {:.4f}, {:.4f}), acm: ({:.4f}, {:.4f}, {:.4f})\n", atom->tag[body[ibody].ilocal], body[ibody].vcm[0], body[ibody].vcm[1], body[ibody].vcm[2], body[ibody].angmom[0], body[ibody].angmom[1], body[ibody].angmom[2]);
    }

  }
}

void FixRigidAbrade::setup_post_neighbor()
{
  
  if ((reinitflag || !setupflag)) {
    // Rebuilding neighbor list following equalise_surface() at the end of the timestep
    if (remesh_flag) end_of_step();
    
    for (int ibody; ibody < (nlocal_body + nghost_body); ibody++) body[ibody].abraded_flag = 0;
    proc_abraded_flag = 0;

  }

  setupflag = 1;

}

/* ----------------------------------------------------------------------
   compute initial fcm and torque on bodies, also initial virial
   reset all particle velocities to be consistent with vcm and omega
------------------------------------------------------------------------- */

void FixRigidAbrade::setup(int vflag)
{

  int i, n, ibody;

  // error if maxextent > comm->cutghost
  // NOTE: could just warn if an override flag set
  // NOTE: this could fail for comm multi mode if user sets a wrong cutoff
  //       for atom types in rigid bodies - need a more careful test
  // must check here, not in init, b/c neigh/comm values set after fix init

  double cutghost = MAX(neighbor->cutneighmax,comm->cutghostuser);
  if (maxextent > cutghost)
    error->all(FLERR,"Rigid body extent {} > ghost atom cutoff - use comm_modify cutoff", maxextent);

  //check(1);

  // sum fcm, torque across all rigid bodies
  // fcm = force on COM
  // torque = torque around COM

  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;

  double *xcm, *fcm, *tcm;
  double dx, dy, dz;

  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    fcm = body[ibody].fcm;
    fcm[0] = fcm[1] = fcm[2] = 0.0;
    tcm = body[ibody].torque;
    tcm[0] = tcm[1] = tcm[2] = 0.0;
  }

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    fcm = b->fcm;
    fcm[0] += f[i][0];
    fcm[1] += f[i][1];
    fcm[2] += f[i][2];

    domain->unmap(x[i], xcmimage[i], unwrap[i]);
    xcm = b->xcm;
    dx = unwrap[i][0] - xcm[0];
    dy = unwrap[i][1] - xcm[1];
    dz = unwrap[i][2] - xcm[2];

    tcm = b->torque;
    tcm[0] += dy * f[i][2] - dz * f[i][1];
    tcm[1] += dz * f[i][0] - dx * f[i][2];
    tcm[2] += dx * f[i][1] - dy * f[i][0];
  }

  // enforce 2d body forces and torques

  if (domain->dimension == 2) enforce2d();

  // reverse communicate fcm, torque of all bodies

  commflag = FORCE_TORQUE;
  comm->reverse_comm(this, 6);

  // virial setup before call to set_v

  v_init(vflag);

  // compute and forward communicate vcm and omega of all bodies

  for (ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];
    MathExtra::angmom_to_omega(b->angmom, b->ex_space, b->ey_space, b->ez_space, b->inertia,
                               b->omega);
  }

  commflag = FINAL;
  comm->forward_comm(this, 10);

  // set velocity/rotation of atoms in rigid bodues

  set_v();

  // guesstimate virial as 2x the set_v contribution

  if (vflag_global)
    for (n = 0; n < 6; n++) virial[n] *= 2.0;
  if (vflag_atom) {
    for (i = 0; i < nlocal; i++)
      for (n = 0; n < 6; n++) vatom[i][n] *= 2.0;
  }

}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::initial_integrate(int vflag)
{
  double dtfm;

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];

    // update vcm by 1/2 step
    dtfm = dtf / b->mass;
    b->vcm[0] += dtfm * b->fcm[0] * dynamic_flag;
    b->vcm[1] += dtfm * b->fcm[1] * dynamic_flag;
    b->vcm[2] += dtfm * b->fcm[2] * dynamic_flag;

    // update xcm by full step

    b->xcm[0] += dtv * b->vcm[0];
    b->xcm[1] += dtv * b->vcm[1];
    b->xcm[2] += dtv * b->vcm[2];

    // update angular momentum by 1/2 step

    b->angmom[0] += dtf * b->torque[0] * dynamic_flag;
    b->angmom[1] += dtf * b->torque[1] * dynamic_flag;
    b->angmom[2] += dtf * b->torque[2] * dynamic_flag;

    // compute omega at 1/2 step from angmom at 1/2 step and current q
    // update quaternion a full step via Richardson iteration
    // returns new normalized quaternion, also updated omega at 1/2 step
    // update ex,ey,ez to reflect new quaternion

    MathExtra::angmom_to_omega(b->angmom, b->ex_space, b->ey_space, b->ez_space, b->inertia,
                               b->omega);
    MathExtra::richardson(b->quat, b->angmom, b->omega, b->inertia, dtq);
    MathExtra::q_to_exyz(b->quat, b->ex_space, b->ey_space, b->ez_space);
  }

  // virial setup before call to set_xv

  v_init(vflag);

  // forward communicate updated info of all bodies

  commflag = INITIAL;
  comm->forward_comm(this, 29);

  // set coords and velocity of atoms in rigid bodies

  set_xv();
}

/* ----------------------------------------------------------------------
   apply Langevin thermostat to all 6 DOF of rigid bodies I own
   unlike fix langevin, this stores extra force in extra arrays,
     which are added in when a new fcm/torque are calculated
------------------------------------------------------------------------- */

void FixRigidAbrade::apply_langevin_thermostat()
{
  double gamma1,gamma2;
  double wbody[3],tbody[3];

  // grow langextra if needed

  if (nlocal_body > maxlang) {
    memory->destroy(langextra);
    maxlang = nlocal_body + nghost_body;
    memory->create(langextra,maxlang,6,"rigid/rigid:langextra");
  }

  double delta = update->ntimestep - update->beginstep;
  delta /= update->endstep - update->beginstep;
  double t_target = t_start + delta * (t_stop-t_start);
  double tsqrt = sqrt(t_target);

  double boltz = force->boltz;
  double dt = update->dt;
  double mvv2e = force->mvv2e;
  double ftm2v = force->ftm2v;

  double *vcm,*omega,*inertia,*ex_space,*ey_space,*ez_space;

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    vcm = body[ibody].vcm;
    omega = body[ibody].omega;
    inertia = body[ibody].inertia;
    ex_space = body[ibody].ex_space;
    ey_space = body[ibody].ey_space;
    ez_space = body[ibody].ez_space;

    gamma1 = -body[ibody].mass / t_period / ftm2v;
    gamma2 = sqrt(body[ibody].mass) * tsqrt *
      sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;
    langextra[ibody][0] = gamma1*vcm[0] + gamma2*(random->uniform()-0.5);
    langextra[ibody][1] = gamma1*vcm[1] + gamma2*(random->uniform()-0.5);
    langextra[ibody][2] = gamma1*vcm[2] + gamma2*(random->uniform()-0.5);

    gamma1 = -1.0 / t_period / ftm2v;
    gamma2 = tsqrt * sqrt(24.0*boltz/t_period/dt/mvv2e) / ftm2v;

    // convert omega from space frame to body frame

    MathExtra::transpose_matvec(ex_space,ey_space,ez_space,omega,wbody);

    // compute langevin torques in the body frame

    tbody[0] = inertia[0]*gamma1*wbody[0] +
      sqrt(inertia[0])*gamma2*(random->uniform()-0.5);
    tbody[1] = inertia[1]*gamma1*wbody[1] +
      sqrt(inertia[1])*gamma2*(random->uniform()-0.5);
    tbody[2] = inertia[2]*gamma1*wbody[2] +
      sqrt(inertia[2])*gamma2*(random->uniform()-0.5);

    // convert langevin torques from body frame back to space frame

    MathExtra::matvec(ex_space,ey_space,ez_space,tbody,&langextra[ibody][3]);
  }
}

/* ----------------------------------------------------------------------
   called from FixEnforce post_force() for 2d problems
   zero all body values that should be zero for 2d model
------------------------------------------------------------------------- */

void FixRigidAbrade::enforce2d()
{
  Body *b;

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    b = &body[ibody];
    b->xcm[2] = 0.0;
    b->vcm[2] = 0.0;
    b->fcm[2] = 0.0;
    b->xgc[2] = 0.0;
    b->torque[0] = 0.0;
    b->torque[1] = 0.0;
    b->angmom[0] = 0.0;
    b->angmom[1] = 0.0;
    b->omega[0] = 0.0;
    b->omega[1] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   Determine if forces imposed on atoms exceeds their hardness. If so, 
   claculate their displacement velocity within their body. 
------------------------------------------------------------------------- */

void FixRigidAbrade::post_force(int /*vflag*/)
{

  if (langflag) apply_langevin_thermostat();
  if (earlyflag) compute_forces_and_torques();

  // update hardness due to variables in input script
  if (varflag != CONSTANT) {
    modify->clearstep_compute();
    if (hstyle == EQUAL) hardness = input->variable->compute_equal(hvar);
    if (mustyle == EQUAL) hardness_ratio = input->variable->compute_equal(muvar);
    if (densitystyle == EQUAL) density = input->variable->compute_equal(densityvar);
    modify->addstep_compute(update->ntimestep + 1);
  }

  // ----------------------------- Calling abrasion functions ---------------------------------------------

  // Atom variables
  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *mask = atom->mask;
  double v_rel[3];
  double x_rel[3];

  // Wall variables
  std::vector<Fix *> wall_list = modify->get_fix_by_style("wall/gran/region");
  double vwall[3];
  double r_contact;
  int nc;

  // variables used to navigate neighbour lists
  int inum = list->inum;
  int *ilist = list->ilist;
  int *numneigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int nlocal = atom->nlocal;

  for (int ibody = nlocal_body; ibody < (nlocal_body + nghost_body); ibody++) {
    body[ibody].wear_energy = 0.0;
  }

  // loop over central atoms in neighbor lists
  for (int ii = 0; ii < inum; ii++) {

    // get local index of central atom ii (i is always a local atom)
    int i = ilist[ii];

    // only process abrasion for atoms in a body
    if (atom2body[i] < 0) continue;

    // check if force is acting on atom i
    // don't need to compute the actual magnitude so forego the sqrt
    if ((f[i][0]*f[i][0]) + (f[i][1]*f[i][1]) + (f[i][2]*f[i][2])) {

      // ~~~~~~~~~~~~~~~~~~~~~~ Processing atom-atom abrasion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

      // get number of neighbours for atom i
      int jnum = numneigh[i];

      // get list of local neighbor ids for atom i
      int *jlist = firstneigh[i];

      // cycle through neighbor list of i and calculate abrasion
      for (int jj = 0; jj < jnum; jj++) {
        // get local index of neighbor to access its properties (j maybe a ghost atom)
        int j = jlist[jj];

        // check that the atoms are not in the same body - using the globally set bodytag[i]
        if (bodytag[i] != bodytag[j]) {

          // Calculating the relative position of i and j in global coordinates
          // Position and velocity of ghost atoms should already be stored for granular simulations
          x_rel[0] = x[j][0] - x[i][0];
          x_rel[1] = x[j][1] - x[i][1];
          x_rel[2] = x[j][2] - x[i][2];

          v_rel[0] = v[j][0] - v[i][0];
          v_rel[1] = v[j][1] - v[i][1];
          v_rel[2] = v[j][2] - v[i][2];

          // Calculate the displacement on atom i from an impact by j
          displacement_of_atom(i, (atom->radius)[j], x_rel, v_rel);
        }
      }

      // ~~~~~~~~~~~~~~~~~~~~~~ Processing atom-wall abrasion ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      
      // Access the region and granular model for each fix_wall/gran/region
      for (auto &fix_region : wall_list) {
        int tmp;
        auto *region = (Region *) fix_region->extract("region", tmp);
        auto *model = (Granular_NS::GranularModel *) fix_region->extract("model", tmp);

        if (!region || !model) {
            error->one(FLERR, "fix rigid/abrade is unable to extract wall contact information from fix wall/gran/region.");
          }
          
        // determine the number of current contacts (nc)
        if (!(mask[i] & groupbit)) continue;
        if (!region->match(x[i][0], x[i][1], x[i][2])) continue;
        nc = region->surface(x[i][0], x[i][1], x[i][2],
                             (atom->radius)[i] + model->pulloff_distance((atom->radius)[i], 0.0));

        // Checking if the region is moving
        int regiondynamic = region->dynamic_check();

        // process current wall contacts for atom i
        for (int ic = 0; ic < nc; ic++) {

          // Setting wall velocity
          if (regiondynamic)
            region->velocity_contact(vwall, x[i], ic);
          else
            vwall[0] = vwall[1] = vwall[2] = 0.0;

          // Setting wall radius at the contact
          // If the curvature is negative treat it as a flat wall
          if (region->contact[ic].radius < 0) {
            r_contact = 0.0;
          } else {
            r_contact = region->contact[ic].radius;
          }

          x_rel[0] = -region->contact[ic].delx;
          x_rel[1] = -region->contact[ic].dely;
          x_rel[2] = -region->contact[ic].delz;

          v_rel[0] = vwall[0] - v[i][0];
          v_rel[1] = vwall[1] - v[i][1];
          v_rel[2] = vwall[2] - v[i][2];

          // Calculate the displacement on atom i from an impact by the wall
          displacement_of_atom(i, r_contact, x_rel, v_rel);
        }
      }
    }
  }  
}

/* ----------------------------------------------------------------------
   Recursively invoke remesh() to remove the most crowded atoms until the 
  standard deviation of surface area per atom is minimised
------------------------------------------------------------------------- */

void FixRigidAbrade::equalise_surface()
{
  
  // Resetting local remesh flags and associated vectors
  proc_remesh_flag = 0;
  global_remesh_flag = 0;
  equalise_surface_flag = 1;
  body_dlist.clear();
  dlist.clear();

  int nlocal = atom->nlocal;

  proc_abraded_flag = 1;
  for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) {

    // Set body abraded flag so that the body is reprocessed in areas_and_normals following remeshing
    body[ibody].abraded_flag = 1;

    // Calculating the average surface area per atom of each body, accounting for the atom placed at the COM which does not have an associated area
    equalise_surface_array[ibody][0] = (body[ibody].surface_area / (body[ibody].natoms - 1));

    // zeroing other elements of the temporary array
    equalise_surface_array[ibody][1] = 0.0;
    equalise_surface_array[ibody][2] = 0.0;
  }

  // Calculating the local Variance
  for (int i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    if (bodytag[i] != atom->tag[i])
      equalise_surface_array[atom2body[i]][1] += (fabs(vertexdata[i][3]) - equalise_surface_array[atom2body[i]][0]) *   (fabs(vertexdata[i][3]) - equalise_surface_array[atom2body[i]][0]);
  }

  // summing variance to owning bodies split across processors
  commflag = EQUALISE;
  comm->reverse_comm(this, 1);

  // communicating this back to the ghost bodies for consistency
  commflag = EQUALISE;
  comm->forward_comm(this, 1);

  // Normalising variance against the average area-per-atom
  for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) {

    //                                           0                  1                     2              3
    // equalise_surface_array[ibody][average_surface_area, sum_area_minus_average_sq, variance_normalised, old variance]
    // Dont actually sqrt to get the variance since this value is only ever compared with itself
    equalise_surface_array[ibody][2] = ((equalise_surface_array[ibody][1]) / (body[ibody].natoms - 1)) / equalise_surface_array[ibody][0];

    if (equalise_surface_array[ibody][2] < equalise_surface_array[ibody][3]) {

      // Setting old variance to current variance for comparision on next call of equalise_surface()
      equalise_surface_array[ibody][3] = equalise_surface_array[ibody][2];

      // only pushback one copy from each unique stored body (this is required for periodic boundaries where a body can be stored as both a local and ghost)
      if (!count(body_dlist.begin(), body_dlist.end(), atom->tag[body[ibody].ilocal])) {
        dlist.push_back(body[ibody].min_area_atom_tag);
        body_dlist.push_back(atom->tag[body[ibody].ilocal]);
      }
    }
  }

  // flagging that a processor is attempting to remove an atom
  if (dlist.size()) proc_remesh_flag = 1;

  // checking if remeshing has been initated on any processor
  MPI_Allreduce(&proc_remesh_flag, &global_remesh_flag, 1, MPI_INT, MPI_SUM, world);

  if (global_remesh_flag) {
    rebuild_flag = 1;
    remesh(dlist);
  }

  // processing the end of the recursive calls of equalise_surface()
  if (equalise_surface_flag) {

    resetup_bodies_static();

    equalise_surface_flag = 0;

    areas_and_normals();

    // Resetting abraded flag for all bodies
    for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) {
      body[ibody].abraded_flag = 0;
    }

    proc_abraded_flag = 0;
  }
}

/* ----------------------------------------------------------------------
  Calcualte the areas and normals of body atoms from their angle topology.
  Also, call remesh() and equalise_surface() if needed
------------------------------------------------------------------------- */

void FixRigidAbrade::areas_and_normals()
{

  int i1, i2, i3;
  double delx1, dely1, delz1, delx2, dely2, delz2;
  double rsq1, rsq2, rsq3, r1, r2;
  double axbi, axbj, axbk, area;
  double sub_area;
  double n1, n2, n3, inverse_length;
  double area_of_atom;

  double **x = atom->x;
  double **f = atom->f;
  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;

  std::vector<std::vector<int>> potential_remeshing_angles;
  std::vector<std::vector<double>> potential_remeshing_angles_normals;

  // Reset vertex data for local and ghost atoms since the ghost atoms may also be accessed when cycling through angles
  // only perform calcualtions on processors which have an abrading body stored locally or as a ghost
  if (proc_abraded_flag) {
      
    for (int i = 0; i < (nlocal + nghost); i++) {
      // Only processing properties relevant to bodies which have abraded and changed shape
      if (atom2body[i] < 0) continue;
      if (!body[atom2body[i]].abraded_flag) continue;
      body[atom2body[i]].body_offset_flag = 1;

      vertexdata[i][0] = 0.0;    // x normal
      vertexdata[i][1] = 0.0;    // y normal
      vertexdata[i][2] = 0.0;    // z normal
      vertexdata[i][3] = 0.0;    // associated area
    }
      
    if (remesh_flag) {
      // Resetting local remesh flags and associated vectors
      global_remesh_flag = 0;
      proc_remesh_flag = 0;
      body_dlist.clear();
      dlist.clear();

      // resetting remesh_atom for local and ghost bodies since only one atom can be removed from a body on a single remesh() call.
      // remesh_atom = 0 if no atom is to be removed
      for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) {
        body[ibody].remesh_atom = 0;
        body[ibody].remesh_atom_displacement_vel = 0.0;
      }
    }

    // cycling through angles (including those from the overflow list if present)
    for (int n = 0; n < (neighbor->nanglelist + remesh_overflow_threshold + overflow_anglelist.size()); n++) {

      // Storing each atom in the angle
      if (n + 1 <= (nanglelist + remesh_overflow_threshold)) {
        i1 = anglelist[n][0];
        i2 = anglelist[n][1];
        i3 = anglelist[n][2];
      } else {
        // reading angles from the overflow list populated during remeshing. This will be cleared and angles properly added to neighbor->anglelist during end_of_step()
        i1 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][0];
        i2 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][1];
        i3 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][2];
      }

      // Only processing properties relevant to bodies which have abraded and changed shape
      if (!body[atom2body[i1]].abraded_flag) continue;

      // 1st edge
      delx1 = displace[i1][0] - displace[i2][0];
      dely1 = displace[i1][1] - displace[i2][1];
      delz1 = displace[i1][2] - displace[i2][2];

      rsq1 = delx1 * delx1 + dely1 * dely1 + delz1 * delz1;
      r1 = sqrt(rsq1);

      // 2nd edge
      delx2 = displace[i3][0] - displace[i2][0];
      dely2 = displace[i3][1] - displace[i2][1];
      delz2 = displace[i3][2] - displace[i2][2];

      rsq2 = delx2 * delx2 + dely2 * dely2 + delz2 * delz2;
      r2 = sqrt(rsq2);

      // cross product
      // a x b = (a2.b3 - a3.b2)i + (a3.b1 - a1.b3)j + (a1.b2 - a2.b1)k
      // b is the first edge whilst a is the second edge.

      axbi = dely2 * delz1 - delz2 * dely1;
      axbj = delz2 * delx1 - delx2 * delz1;
      axbk = delx2 * dely1 - dely2 * delx1;

      // area of angle
      area = (0.5) * sqrt(axbi * axbi + axbj * axbj + axbk * axbk);
      sub_area = area * (THIRD);

      n1 = axbi;
      n2 = axbj;
      n3 = axbk;

      // distributing facet areas and normals to the vertex atoms
      vertexdata[i1][3] += sub_area;
      vertexdata[i1][0] += n1;
      vertexdata[i1][1] += n2;
      vertexdata[i1][2] += n3;

      vertexdata[i2][3] += sub_area;
      vertexdata[i2][0] += n1;
      vertexdata[i2][1] += n2;
      vertexdata[i2][2] += n3;

      vertexdata[i3][3] += sub_area;
      vertexdata[i3][0] += n1;
      vertexdata[i3][1] += n2;
      vertexdata[i3][2] += n3;
    
      // constructing a list of angles that contain abrading atoms that can later be searched for remeshing. Also cache their normals so they do not need to be recomputed.
      // additionally store the angle if any of the locally stored angles contain an atom whose area is below the remeshing threhsold
      // if any of the vertex atoms have a displacement velocity, the angle has abraded
      if (check_threshold_flag &&
        ( vertexdata[i1][4] || vertexdata[i1][5] || vertexdata[i1][6] || 
          vertexdata[i2][4] || vertexdata[i2][5] || vertexdata[i2][6] || 
          vertexdata[i3][4] || vertexdata[i3][5] || vertexdata[i3][6] )) {

            potential_remeshing_angles.push_back({i1,i2,i3});
            potential_remeshing_angles_normals.push_back({n1,n2,n3});
          }
    }
  }

  // reverse communicate contribution to normals of ghost atoms
  // This is required for particles that straddle domain boundaries
  // all procs, even those without an abrading atom, must communicate
  commflag = NORMALS;
  comm->reverse_comm(this, 4);
  

  
  if (proc_abraded_flag){
    // Setting the surface area and other remeshing properties of all abraded bodies (ghost and local) to 0 if the remesh flag is set
    for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) {
      
      // Only processing properties relevant to bodies which have abraded and changed shape
      if (!body[ibody].abraded_flag) continue;
      
      body[ibody].surface_area = 0.0;
      
        if (remesh_flag) {  
        // intialising variables to find the minimum area atom for each body
        body[ibody].min_area_atom = BIG;
        body[ibody].min_area_atom_tag = 0;
      }
    }
    
    // normalise the length of all atom normals
    for (int i = 0; i < nlocal; i++) {

      // Only processing properties relevant to bodies which have abraded and changed shape
      if (atom2body[i] < 0) continue;
      if (!body[atom2body[i]].abraded_flag) continue;

      n1 = vertexdata[i][0];
      n2 = vertexdata[i][1];
      n3 = vertexdata[i][2];
      inverse_length = 1.0 / sqrt(n1 * n1 + n2 * n2 + n3 * n3);

      Body *b = &body[atom2body[i]];

      // Storing normals in body coordinates
      vertexdata[i][0] = n1 *= inverse_length;
      vertexdata[i][1] = n2 *= inverse_length;
      vertexdata[i][2] = n3 *= inverse_length;

      // calculating surface area of each body
      if ((bodytag[i] != atom->tag[i])) {

        area_of_atom = fabs(vertexdata[i][3]);

        b->surface_area += area_of_atom;

        // Storing the bodies' atom with the minimum associated area if the remesh flag is set
        if (remesh_flag) {
          if (area_of_atom <= (b->min_area_atom * (1.0 + 0.001))) {
            // if the incoming area is equal to the current stored area then give precident to the larger atom tag
            if (area_of_atom < (b->min_area_atom * (1.0 - 0.001))) {
              b->min_area_atom = area_of_atom;
              b->min_area_atom_tag = atom->tag[i];
            } else {
              if (atom->tag[i] > b->min_area_atom_tag) {
                b->min_area_atom = area_of_atom;
                b->min_area_atom_tag = atom->tag[i];
              }
            }
          }
        }
      }
    }
  }
  
  if (remesh_flag) {
    commflag = NORMALS;
    comm->forward_comm(this,4);

    double dot_prod_i1, dot_prod_i2, dot_prod_i3;
    double displace_vel_i1, displace_vel_i2, displace_vel_i3;
    int remove_id;
    int is_flipped;
    for (int n = 0; n < potential_remeshing_angles.size(); n++) {
      
      i1 = potential_remeshing_angles[n][0];
      i2 = potential_remeshing_angles[n][1];
      i3 = potential_remeshing_angles[n][2];
      
      n1 = potential_remeshing_angles_normals[n][0];
      n2 = potential_remeshing_angles_normals[n][1];
      n3 = potential_remeshing_angles_normals[n][2];
      
      // normalising so that the dot product with the vertex normals is bounded between -1 and 1 with the magnitude indicating the relative allignment
      inverse_length = 1.0 / sqrt(n1*n1 + n2*n2 + n3*n3);
      
      n1 *= inverse_length;
      n2 *= inverse_length;
      n3 *= inverse_length;

      // First check if the facets normal is in the opposite direction to the normals of one of its  vertexes indicating that a facet has flipped
      is_flipped = 0;
      
      dot_prod_i1 = (vertexdata[i1][0] * n1) + (vertexdata[i1][1] * n2) + (vertexdata[i1][2] * n3);
      dot_prod_i2 = (vertexdata[i2][0] * n1) + (vertexdata[i2][1] * n2) + (vertexdata[i2][2] * n3);
      dot_prod_i3 = (vertexdata[i3][0] * n1) + (vertexdata[i3][1] * n2) + (vertexdata[i3][2] * n3);
      
      is_flipped = (dot_prod_i1 <= 0.0) || (dot_prod_i2 <= 0.0) || (dot_prod_i3 <= 0.0);

      if (!is_flipped)
        continue;
      
      remove_id = -1;
      
      displace_vel_i1 = sqrt(vertexdata[i1][4]*vertexdata[i1][4] + vertexdata[i1][5]*vertexdata[i1][5] + vertexdata[i1][6]*vertexdata[i1][6]);
      displace_vel_i2 = sqrt(vertexdata[i2][4]*vertexdata[i2][4] + vertexdata[i2][5]*vertexdata[i2][5] + vertexdata[i2][6]*vertexdata[i2][6]);
      displace_vel_i3 = sqrt(vertexdata[i3][4]*vertexdata[i3][4] + vertexdata[i3][5]*vertexdata[i3][5] + vertexdata[i3][6]*vertexdata[i3][6]);

        // only considering vertexes which have abraded and only if it is has the largest displacement velocity in any flipped facets in the body
        if (displace_vel_i1 > body[atom2body[i1]].remesh_atom_displacement_vel){
          body[atom2body[i1]].remesh_atom_displacement_vel = displace_vel_i1;
          remove_id = i1;
        }


        // repeating for i2 
        if (displace_vel_i2 > body[atom2body[i2]].remesh_atom_displacement_vel){
          body[atom2body[i2]].remesh_atom_displacement_vel = displace_vel_i2;
          remove_id = i2;
        }


        // repeating for i3
        if (displace_vel_i3 > body[atom2body[i3]].remesh_atom_displacement_vel){
          body[atom2body[i3]].remesh_atom_displacement_vel = displace_vel_i3;
          remove_id = i3;
        }

        if (remove_id >= 0)
          body[atom2body[remove_id]].remesh_atom = atom->tag[remove_id];
      }
      
    potential_remeshing_angles.clear();
    potential_remeshing_angles_normals.clear();
  }

  // Reverse comminicate the min area atom from the ghost bodies back to the locally stored body.
  // Instead of simply summing,overide the min area if it is smaller than the one stored locally
  // also include the body surface area aswell to any remesh flags set from inverted angles

  if (!remesh_flag) {
    commflag = SURFACE_AREA;
    comm->reverse_comm(this, 1);
    commflag = SURFACE_AREA;
    comm->forward_comm(this, 1);
  }
  
  else {
    commflag = MIN_AREA;
    comm->reverse_comm(this, 5);
    commflag = MIN_AREA;
    comm->forward_comm(this, 5);

    if (proc_abraded_flag)
      for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) {

        if (!body[ibody].abraded_flag) continue;

        if (check_threshold_flag && (!body[ibody].remesh_atom) && (body[ibody].min_area_atom <  (remesh_area_threshold_atom * remesh_threshold_multiplier_atom))) {
          body[ibody].remesh_atom = body[ibody].min_area_atom_tag;
        }

        // If the body hasn't been remeshed set its abraded flag to 0 so it is not considered when reprocessing the area_and_normals on the current timestep
        if (!body[ibody].remesh_atom) {
          body[ibody].abraded_flag = 0;
        } else {
          // pushback one copy from each unique stored body (this is required for periodic boundaries where a body can be stored as both a local and ghost)
          if (!count(body_dlist.begin(), body_dlist.end(), atom->tag[body[ibody].ilocal])) {
            dlist.push_back(body[ibody].remesh_atom);
            body_dlist.push_back(atom->tag[body[ibody].ilocal]);
          }
        }
      }

      // flagging that a processor is attempting to remove an atom
      if (dlist.size()) proc_remesh_flag = 1;

      // checking if remeshing has been initated on any processor
      MPI_Allreduce(&proc_remesh_flag, &global_remesh_flag, 1, MPI_INT, MPI_SUM, world);

      if (global_remesh_flag) {
        rebuild_flag = 1;
        remesh(dlist);
      }

    if (equalise_surface_flag) equalise_surface();
  }
}

/* ----------------------------------------------------------------------
  Calculate the displacement velocity of an abrading atom in the global
  coordinate system
------------------------------------------------------------------------- */

void FixRigidAbrade::displacement_of_atom(int i, double impacting_radius, double x_rel[3],
                                          double v_rel[3])
{
  double **f = atom->f;
  double **x = atom->x;
  double **v = atom->v;
  double dt = update->dt;
  
  double r_eff;
  double bodynormals[3];
  double global_normals[3];
  bool gap_is_shrinking;
  double associated_area;
  double f_norm_dot;
  double f_norm[3];
  double f_tan[3];
  bool indentation;
  bool scratching;
  double v_norm_dot;
  double v_norm[3];
  double v_tan[3];
  double norm_disp_vel[3];
  double htp;
  double tan_disp_vel_abs;
  double tan_disp_vel[3];
  double total_disp_vel[3];
  

  // Calculating the effective radius of atom i and the impacting body (atom j or wall)
  r_eff = (atom->radius)[i] + impacting_radius;

  // Check that the i and the impactor are in contact - This is necessary as otherwise atoms which are aproaching i can contribute to its abrasion even if they are on the other side of the simulation
  if (MathExtra::len3(x_rel) > r_eff) return;

  // Converting normals from body coordinates to global coordinates so they can be compared with atom velocities
  bodynormals[0] = vertexdata[i][0];
  bodynormals[1] = vertexdata[i][1];
  bodynormals[2] = vertexdata[i][2];

  Body *b = &body[atom2body[i]];
  MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, bodynormals, global_normals);

  // Checking if the normal and relative velocities are facing towards oneanother,
  // indicating the gap between i and j is closing
  gap_is_shrinking = (MathExtra::dot3(v_rel, global_normals) < 0);

  // If the atoms are moving away from one another flip the sign of v_rel such
  //  that the calcualted displacement still acts inwards along the normal
  if (!gap_is_shrinking) {
    v_rel[0] = -v_rel[0];
    v_rel[1] = -v_rel[1];
    v_rel[2] = -v_rel[2];
  }

  // Storing area associated with atom i (equal in both body and global coords)
  associated_area = vertexdata[i][3];

  // Calculating force acting along the surface atom's normal
  f_norm_dot = f[i][0] * global_normals[0] + f[i][1] * global_normals[1] + f[i][2] * global_normals[2];
  f_norm[0] = f_norm_dot * global_normals[0];
  f_norm[1] = f_norm_dot * global_normals[1];
  f_norm[2] = f_norm_dot * global_normals[2];

  // Calculating force acting tangential the surface atoms normal
  f_tan[0] = f[i][0] - f_norm[0];
  f_tan[1] = f[i][1] - f_norm[1];
  f_tan[2] = f[i][2] - f_norm[2];

  // Comparing normal stress against hardness to determine if indentation is occuring
  double f_norm_magnitude = MathExtra::len3(f_norm);
  indentation = (f_norm_magnitude / associated_area) > hardness;

  // Comparing shear stress against hardness to determine if scratching is occuring
  scratching = (MathExtra::len3(f_tan) / associated_area) > (hardness*hardness_ratio);

  // if no abrasion is occuring return from function with no displacement velocity
  if (!indentation && !scratching)
    return;

  // Calculating relative velocity acting along the surface atom's normal
  v_norm_dot = v_rel[0] * global_normals[0] + v_rel[1] * global_normals[1] + v_rel[2] * global_normals[2];
  v_norm[0] = v_norm_dot * global_normals[0];
  v_norm[1] = v_norm_dot * global_normals[1];
  v_norm[2] = v_norm_dot * global_normals[2];

  // Normal displacement velocity is equal to v_norm 
  norm_disp_vel[0] = v_norm[0];
  norm_disp_vel[1] = v_norm[1];
  norm_disp_vel[2] = v_norm[2];

  // Tangential displacement velocity needs to be calculated from geometric considerations
  // such that it is still displaced along the atom's normal

  // Calculating velocity acting tangential the surface atom's normal
  v_tan[0] = v_rel[0] - v_norm[0];
  v_tan[1] = v_rel[1] - v_norm[1];
  v_tan[2] = v_rel[2] - v_norm[2];
  // calculating the per-atom indentation depth 
  htp = r_eff - (x_rel[0]*global_normals[0] + x_rel[1]*global_normals[1] + x_rel[2]*global_normals[2]);

  // htp this must be positive for scratching to occur
  if (htp > 0.0) {
    tan_disp_vel_abs = - (htp - r_eff + sqrt( (r_eff - htp)*(r_eff - htp) + 2.0 * sqrt(2.0*r_eff*htp - (htp*htp))*( MathExtra::len3(v_tan) * dt))) / dt;
    tan_disp_vel[0] = tan_disp_vel_abs * global_normals[0];
    tan_disp_vel[1] = tan_disp_vel_abs * global_normals[1];
    tan_disp_vel[2] = tan_disp_vel_abs * global_normals[2];
  }
  else {
    tan_disp_vel[0] = 0.0;
    tan_disp_vel[1] = 0.0;
    tan_disp_vel[2] = 0.0;
  }
  
  // Assign displacement velocities, alligned normally, to atom i in global coordinates
  // Cylcing over all neighbours to i means that each interaction can contribute independently to the abrasion of i
  
  total_disp_vel[0] =  (indentation * norm_disp_vel[0]) + (scratching * tan_disp_vel[0]);
  total_disp_vel[1] =  (indentation * norm_disp_vel[1]) + (scratching * tan_disp_vel[1]);
  total_disp_vel[2] =  (indentation * norm_disp_vel[2]) + (scratching * tan_disp_vel[2]);
  
  vertexdata[i][4] += total_disp_vel[0];
  vertexdata[i][5] += total_disp_vel[1];
  vertexdata[i][6] += total_disp_vel[2];
  
  // Calculating the cumulative work done to displace the atom
  // vertexdata[i][8] += fabs(fnorm) * fabs(total_normal_displacement);

  // Calculating the cumulative work done to displace the atom along the normal (Work = Force * Distance)
  // Although the atom is locally stored, it could belong to a ghost body
  body[atom2body[i]].wear_energy += f_norm_magnitude * MathExtra::len3(total_disp_vel)*update->dt;
}

/* ----------------------------------------------------------------------
 Function to check if an angle proposed in remesh() is valid. An angle is assumed valid if:
  - It is defined in a CCW direction in the projected coordinate system 
  - An reflex vertex in the boundary is not contatined within the proposed angle in the projected coordinates
---------------------------------------------------------------------- */
bool FixRigidAbrade::valid_angle_check(int a_index, int b_index, int c_index, const std::vector<std::vector<double>>  &boundary_projected, const std::vector<double>  &interior_thetas)
{
   bool is_valid_angle = 1;

    // storing  angle coords from the projected boundary
   double a[2] = {boundary_projected[a_index][0], boundary_projected[a_index][1]};
   double b[2] = {boundary_projected[b_index][0], boundary_projected[b_index][1]};
   double c[2] = {boundary_projected[c_index][0], boundary_projected[c_index][1]};

    // defining vectors from central atom
    double bc[2] = {(c[0]-b[0]) , (c[1]-b[1])};
    double ba[2] = {(a[0]-b[0]) , (a[1]-b[1])};

    // check that the angle is defined in a CCW direction
    double bc_cross_ba = (bc[0]*ba[1] - bc[1]*ba[0]);
    if (bc_cross_ba < 0.0) {
        is_valid_angle = 0;
        return(is_valid_angle);
    }

    // cycle through points again, ignoring both the ones in the proposed angle, and any concave points
    for (int j = 0; j < boundary_projected.size(); j++) {
        
        if (j == a_index || j == b_index || j == c_index) 
            continue;

        // only check reflex vertices:
        if (interior_thetas[j] < 180.0) 
            continue;
        
        // if P lays inside the proposed angle then is not valid
        double P[2] = {boundary_projected[j][0], boundary_projected[j][1]};

        double barycentric_denom = ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]));
        if( barycentric_denom == 0.0){
            is_valid_angle = 0;
            break;
        }

        double barycentric_a = ((b[1] - c[1]) * (P[0] - c[0]) + (c[0] - b[0]) * (P[1] - c[1])) / barycentric_denom;
        double barycentric_b = ((c[1] - a[1]) * (P[0] - c[0]) + (a[0] - c[0]) * (P[1] - c[1])) / barycentric_denom;
        double barycentric_c = 1 - barycentric_a - barycentric_b;
        if (0.0 <= barycentric_a && barycentric_a <= 1.0 && 0.0 <= barycentric_b && barycentric_b <= 1.0 && 0.0 <= barycentric_c && barycentric_c <= 1.0){
            is_valid_angle = 0;
            break;
        }
        
    }
    
    return is_valid_angle;   

}

/* ----------------------------------------------------------------------
 Function to return the internal thetas of an angle defined by vertex 
 indexes into a remeshing boundary
---------------------------------------------------------------------- */

void FixRigidAbrade::triangle_thetas(int a_index, int b_index, int c_index, const std::vector<std::vector<double>>  &boundary_projected, double *ans){

    // storing  angle coords from the projected boundary
    double a[2] = {boundary_projected[a_index][0], boundary_projected[a_index][1]};
    double b[2] = {boundary_projected[b_index][0], boundary_projected[b_index][1]};
    double c[2] = {boundary_projected[c_index][0], boundary_projected[c_index][1]};

    // theta at a:
    double ab[2] = {(b[0]-a[0]) , (b[1]-a[1])};
    double ac[2] = {(c[0]-a[0]) , (c[1]-a[1])};
    ans[0] = acos(  (ab[0]*ac[0]+ab[1]*ac[1])   /   (   sqrt(ab[0]*ab[0]+ab[1]*ab[1])   *   sqrt(ac[0]*ac[0]+ac[1]*ac[1]) ) ) * (180.0/M_PI);

    // theta at b:
    double bc[2] = {(c[0]-b[0]) , (c[1]-b[1])};
    double ba[2] = {(a[0]-b[0]) , (a[1]-b[1])};
    ans[1] = acos(  (bc[0]*ba[0]+bc[1]*ba[1])   /   (   sqrt(bc[0]*bc[0]+bc[1]*bc[1])   *   sqrt(ba[0]*ba[0]+ba[1]*ba[1]) ) ) * (180.0/M_PI);

    // theta at c:
    double ca[2] = {(a[0]-c[0]) , (a[1]-c[1])};
    double cb[2] = {(b[0]-c[0]) , (b[1]-c[1])};
    ans[2] = acos(  (ca[0]*cb[0]+ca[1]*cb[1])   /   (   sqrt(ca[0]*ca[0]+ca[1]*ca[1])   *   sqrt(cb[0]*cb[0]+cb[1]*cb[1]) ) ) * (180.0/M_PI);
}


/* ----------------------------------------------------------------------
  Offsetting the positions of atom inwards towards the COM by by their radius.
  ---------------------------------------------------------------------- */
void FixRigidAbrade::offset_vertices_inwards(){

  if (!offset_flag) return;

  int nlocal = atom->nlocal;
  double *radius = atom->radius;
  double **x = atom->x;
  double len_i;

  int xbox, ybox, zbox;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;
  
  for (int i = 0; i < nlocal; i++){
    
    // Only processing properties relevant to bodies which have abraded and changed shape
    if (atom2body[i] < 0) continue;
    if (!body[atom2body[i]].body_offset_flag) continue;
    if (bodytag[i] == atom->tag[i]) continue;

    Body *b = &body[atom2body[i]];

    // offsetting atoms outwards towards the COM by the atoms radius  
    len_i = MathExtra::len3(displace[i]);
    displace[i][0] = displace[i][0] * (1.0 - radius[i]/len_i);
    displace[i][1] = displace[i][1] * (1.0 - radius[i]/len_i);
    displace[i][2] = displace[i][2] * (1.0 - radius[i]/len_i);

    MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, displace[i], x[i]);

    // This transormation is with respect to (0,0,0) in the global coordinate space, need to translate the position of atom i by the postition of its body's COM
    // Additionally, map back into periodic box via xbox,ybox,zbox
    // same for triclinic, add in box tilt factors as well

    xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
    ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;

    // add center of mass to displacement
    if (triclinic == 0) {
      x[i][0] += b->xcm[0] - xbox * xprd;
      x[i][1] += b->xcm[1] - ybox * yprd;
      x[i][2] += b->xcm[2] - zbox * zprd;
    } else {
      x[i][0] += b->xcm[0] - xbox * xprd - ybox * xy - zbox * xz;
      x[i][1] += b->xcm[1] - ybox * yprd - zbox * yz;
      x[i][2] += b->xcm[2] - zbox * zprd;
    }
  }

  for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) {
    if (body[ibody].body_offset_flag){
      body[ibody].abraded_flag = 1;
      body[ibody].body_offset_flag = 0;
    }
  }

  proc_abraded_flag = 1;
  
}

/* ----------------------------------------------------------------------
  Offsetting the positions of atom outwards away from the COM by by their radius.
  ---------------------------------------------------------------------- */
void FixRigidAbrade::offset_vertices_outwards(){

  if (!offset_flag) return;

  int nlocal = atom->nlocal;
  double *radius = atom->radius;
  double **x = atom->x;
  double len_i;

  int xbox, ybox, zbox;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;


  for (int i = 0; i < nlocal; i++){

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (atom2body[i] < 0) continue;
    if (!body[atom2body[i]].abraded_flag) continue;
    if (bodytag[i] == atom->tag[i]) continue;

    Body *b = &body[atom2body[i]];

    // offsetting atoms outwards towards the COM by the atoms radius  
    len_i = MathExtra::len3(displace[i]);
    displace[i][0] = displace[i][0] * (1.0 + radius[i]/len_i);
    displace[i][1] = displace[i][1] * (1.0 + radius[i]/len_i);
    displace[i][2] = displace[i][2] * (1.0 + radius[i]/len_i);


    MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, displace[i], x[i]);

    // This transormation is with respect to (0,0,0) in the global coordinate space, need to translate the position of atom i by the postition of its body's COM
    // Additionally, map back into periodic box via xbox,ybox,zbox
    // same for triclinic, add in box tilt factors as well

    xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
    ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;

    // add center of mass to displacement
    if (triclinic == 0) {
      x[i][0] += b->xcm[0] - xbox * xprd;
      x[i][1] += b->xcm[1] - ybox * yprd;
      x[i][2] += b->xcm[2] - zbox * zprd;
    } else {
      x[i][0] += b->xcm[0] - xbox * xprd - ybox * xy - zbox * xz;
      x[i][1] += b->xcm[1] - ybox * yprd - zbox * yz;
      x[i][2] += b->xcm[2] - zbox * zprd;
    }
  }
}

/* ----------------------------------------------------------------------
    Function adapted from fix_bond_break.cpp to remove a dlist of atoms
    from owned bodies and rebuild the local topology into overflow lists.
---------------------------------------------------------------------- */

void FixRigidAbrade::remesh(std::vector<int> dlist)
{

  // Variable to store the current atom being removed and the tag of the atom owning its body
  int remove_tag = 0;
  int body_remove_tag = 0;

  // clearing remeshing vectors and setting flags
  boundaries.clear();
  edges.clear();
  new_angles_list.clear();
  new_angles_type.clear();
  remesh_overflow_threshold = 0;
  remesh_angle_proc_difference = 0;

  // Cycling through atoms to be removed to set their appropriate tags and decrement the number of atoms in their respective bodies
  for (int dlist_i = 0; dlist_i < dlist.size(); dlist_i++) {

    // initiate vector to hold the boundaries and edges for each atom in dlist (intiated now to preserve indexing used in communication)
    boundaries.push_back({});
    edges.push_back({});
    new_angles_list.push_back({});
    new_angles_type.push_back(-1);    // -1 being a placeholder value which will be updated during remeshing

    // Reading in the tag of the atom in the dlist and its owning body
    remove_tag = dlist[dlist_i];
    body_remove_tag = body_dlist[dlist_i];

    // storing all the atoms that have been remeshed on this timestep so that they can be deleted before the neighborlists are rebuilt.
    // This is a separate vector to dlist since multiple dlists can be remeshed on a given timestep under a single rebuilding of the neighbor lists.
    total_dlist.push_back(remove_tag);
    total_body_dlist.push_back(body_remove_tag);

    // Get the local index of the atom to be removed and the index of the owning body
    tagint remove_id = atom->map(remove_tag);
    tagint body_remove_id = atom2body[atom->map(body_remove_tag)];
        
    // decrementing the number of atoms in the body
    body[body_remove_id].natoms--;

    // checking that there are enough atoms remaining to form a 3D closed boundary
    if (body[body_remove_id].natoms < 5) {

      error->one(FLERR,
                 "fix rigid/abrade has attempted to remesh a body ({}) with less than 5 atoms. ",
                 body_remove_tag);
    }

    // Set abraded flag of body of removed atom so that it considered in subsequent areas_and_normals() calls
    body[body_remove_id].abraded_flag = 1;

    // If the atom is stored on the processors then set its associated body flags to mark that the it no longer belongs to a body
    // It is therefore not processed in areas_and_normals() and resetup_bodies_static()
    // This prevents the need to repeatedly rebuild the neighbor lists each time an atom is removed during remesh().

    if (remove_id >= 0) {
      if (atom2body[remove_id] < 0)
        error->one(FLERR,
                   "Fix Rigid/Abrade has attempted to remesh an atom {} that does not belong to a "
                   "rigid body",
                   remove_tag);
      bodytag[remove_id] = 0;
      atom2body[remove_id] = -1;
    }
  }

  // ------------------------------------------- Breaking angles and constructing edges around the remesh boundary ------------------------------------------

  int nlocal = atom->nlocal;

  // Break any angle which contains a target atom in dlist and store the type of angles used
  int remesh_atype = -1;
  for (int m = 0; m < nlocal; m++) {
    int j, found;

    int num_angle = atom->num_angle[m];
    int *angle_type = atom->angle_type[m];
    tagint *angle_atom1 = atom->angle_atom1[m];
    tagint *angle_atom2 = atom->angle_atom2[m];
    tagint *angle_atom3 = atom->angle_atom3[m];

    int i = 0;

    // Cycle through angles
    while (i < num_angle) {
      found = 0;

      // checking if atom1, atom2, or atom3 is in the dlist
      if (count(dlist.begin(), dlist.end(), angle_atom1[i]) ||
          count(dlist.begin(), dlist.end(), angle_atom2[i]) ||
          count(dlist.begin(), dlist.end(), angle_atom3[i])) {
        found = 1;
        remesh_atype = angle_type[i];

        // constructing edges in an counter-clockwise (CCW) manner and store the angle type with that dlist atom. Only one angle type can be stored with each dlist atom.
        if (count(dlist.begin(), dlist.end(), angle_atom1[i])) {
          edges[static_cast<int>(find(dlist.begin(), dlist.end(), angle_atom1[i]) - dlist.begin())]
              .push_back({angle_atom2[i], angle_atom3[i]});
          new_angles_type[static_cast<int>(find(dlist.begin(), dlist.end(), angle_atom1[i]) -
                                           dlist.begin())] = remesh_atype;
        }
        if (count(dlist.begin(), dlist.end(), angle_atom2[i])) {
          edges[static_cast<int>(find(dlist.begin(), dlist.end(), angle_atom2[i]) - dlist.begin())]
              .push_back({angle_atom3[i], angle_atom1[i]});
          new_angles_type[static_cast<int>(find(dlist.begin(), dlist.end(), angle_atom2[i]) -
                                           dlist.begin())] = remesh_atype;
        }
        if (count(dlist.begin(), dlist.end(), angle_atom3[i])) {
          edges[static_cast<int>(find(dlist.begin(), dlist.end(), angle_atom3[i]) - dlist.begin())]
              .push_back({angle_atom1[i], angle_atom2[i]});
          new_angles_type[static_cast<int>(find(dlist.begin(), dlist.end(), angle_atom3[i]) -
                                           dlist.begin())] = remesh_atype;
        }
      }

      // Unassign angles so they aren't included in the locally rebuilt topology
      if (!found)
        i++;
      else {
        for (j = i; j < num_angle - 1; j++) {
          angle_type[j] = angle_type[j + 1];
          angle_atom1[j] = angle_atom1[j + 1];
          angle_atom2[j] = angle_atom2[j + 1];
          angle_atom3[j] = angle_atom3[j + 1];
        }
        num_angle--;
      }
    }

    atom->num_angle[m] = num_angle;
  }

  // Communicate these locally contructed edges back to the proc which owns the dlist atom
  // Also include the nangle types of the broken angles to be later forward communicated back to ghost bodies in NEW_ANGLES

  commflag = EDGES;
  comm->reverse_comm(this);

  // Construct the boundary for each deleted atom in a CCW direction from the correspoinding edges (only on the proc which owns it)

  for (int edge = 0; edge < edges.size(); edge++) {

    if (edges[edge].size()) {

      int count = 2;
      (boundaries[edge]).push_back(edges[edge][0][0]);
      (boundaries[edge]).push_back(edges[edge][0][1]);
      tagint current_atom = edges[edge][0][1];

      while (count < edges[edge].size()) {
        for (int i = 0; i < edges[edge].size(); i++) {
          if (edges[edge][i][0] == current_atom) {
            current_atom = edges[edge][i][1];
            (boundaries[edge]).push_back(current_atom);
            count++;
            if (count == edges[edge].size()) break;
          }
        }
      }
    }
  }


  for (int dlist_i = 0; dlist_i < dlist.size(); dlist_i++) {

    remove_tag = dlist[dlist_i];

    // only generate new angles on the proc that owns the target atom
    if ((atom->map(remove_tag) < 0) || (atom->map(remove_tag) >= nlocal)) continue;

    // only considering locally owned atoms from here on

    // Getting the boundary for that atom
    std::vector<tagint> boundary = boundaries[dlist_i];
    std::vector<tagint> temp_boundary;

    // rearranging the boundary so that the first atom tag is the maximum (to create consistency across MPI processes)

    // finding index of max value in boundary.
    int max_tag_index = static_cast<int>(
        find(boundary.begin(), boundary.end(), (*max_element(boundary.begin(), boundary.end()))) -
        boundary.begin());

    // store boundary and count the number of locally stored boundary atoms
    double n_local_boundary = 0.0;
    
    for (int i = 0; i < boundary.size(); i++){
      
      if (atom->map(boundary[i]) >= 0 && atom->map(boundary[i]) < atom->nlocal) 
        n_local_boundary += 1.0;

      temp_boundary.push_back(boundary[(i + max_tag_index) % static_cast<int>(boundary.size())]);
    }
    boundary = temp_boundary;
    
    std::vector<std::vector<double>> boundary_coords;

    // Popoulating boundary_coords with the displacements of each atom in the boundary
    for (int boundary_point = 0; boundary_point < (boundary).size(); boundary_point++) {
      boundary_coords.push_back({displace[atom->map((boundary)[boundary_point])][0],
                                 displace[atom->map((boundary)[boundary_point])][1],
                                 displace[atom->map((boundary)[boundary_point])][2]});
    }

    // Storing target atom's position in its body coordinates
    double removed_atom_position[3];
    removed_atom_position[0] = displace[atom->map(remove_tag)][0];
    removed_atom_position[1] = displace[atom->map(remove_tag)][1];
    removed_atom_position[2] = displace[atom->map(remove_tag)][2];

    // Storing target atom's normal in its body coordinates
    double removed_atom_normal[3];
    removed_atom_normal[0] = vertexdata[atom->map(remove_tag)][0];
    removed_atom_normal[1] = vertexdata[atom->map(remove_tag)][1];
    removed_atom_normal[2] = vertexdata[atom->map(remove_tag)][2];

    
    // recess the reference origin, used to project the boundary, into the surface by the dlist atom's normal
    double origin[3] = {removed_atom_position[0] - removed_atom_normal[0], removed_atom_position[1] - removed_atom_normal[1], removed_atom_position[2] - removed_atom_normal[2]};

    // the vector pointing out of the plane is therefore just the dlist atom's normal
    double origin_to_removed_atom[3] = {removed_atom_normal[0], removed_atom_normal[1], removed_atom_normal[2]};

    // Projecting the boundary onto a plane defined by the vector connecting the reference origin to the removed atom
    // Equation of the plane defined by the removed atom and its normal
    // ax + by + cz = d
    double a_plane = origin_to_removed_atom[0];
    double b_plane = origin_to_removed_atom[1];
    double c_plane = origin_to_removed_atom[2];
    double d_plane = MathExtra::dot3(removed_atom_position, origin_to_removed_atom);

    std::vector<std::vector<double>> boundary_projected;

    for (int i = 0; i < boundary_coords.size(); i++) {
      double *point = &boundary_coords[i][0];
      double k = d_plane - a_plane * point[0] - b_plane * point[1] - c_plane * point[2];
      boundary_projected.push_back(
          {(point[0] + k * a_plane), (point[1] + k * b_plane), (point[2] + k * c_plane)});
    }

    std::vector<std::vector<double>> boundary_projected_2d;
    //  deining new axis connecting the removed atom to the first point in the boundary
    double e1[3];
    double *first_point = &boundary_projected[0][0];
    MathExtra::sub3(removed_atom_position, first_point, e1);
    e1[0] = e1[0] / MathExtra::len3(e1);
    e1[1] = e1[1] / MathExtra::len3(e1);
    e1[2] = e1[2] / MathExtra::len3(e1);
    // second axis is at a right angle alligned on the plane
    double e2[3];
    MathExtra::cross3(origin_to_removed_atom, e1, e2);
    e2[0] = e2[0] / MathExtra::len3(e2);
    e2[1] = e2[1] / MathExtra::len3(e2);
    e2[2] = e2[2] / MathExtra::len3(e2);

    for (int i = 0; i < boundary_projected.size(); i++) {
      double *point = &boundary_projected[i][0];
      boundary_projected_2d.push_back(
          {MathExtra::dot3(point, e1), MathExtra::dot3(point, e2), 0.0});
    }

    boundary_projected = boundary_projected_2d;

    // Start of remeshing scheme adapted from G. Mei, J. C. Tipper, N. Xu, Ear-Clipping Based Algorithms of Generating High-Quality Polygon Triangulation, in: W. Lu, G. Cai, W. Liu, W. Xing (Eds.), Proceedings of the 2012 International Conference on Information Technology and Software Engineering, Springer, Berlin, Heidelberg, 2013, pp. 979–988. doi:10.1007/978-3-642-34531-9 105.

   // create empty vector of size boundary_projected.size() to hold the internal angles
    std::vector<double> interior_thetas;
    for (int i = 0; i < boundary_projected.size(); i++) {
        interior_thetas.push_back(0.0);
    }

    std::vector<int> masked_atoms;
    std::vector<std::vector<int>> remeshed_angles;

    while (remeshed_angles.size() < (boundary_projected.size()-2)) {

        // cycle around the boundary and store internal angles. Mark convex if theta < 180 deg, otherwise mark reflex
        for (int i = 0; i < boundary_projected.size(); i++) {

            // getting the indexes of the proposed angle, skipping over the masked atoms that have already had angles defined for them
            int a_index, b_index, c_index;

            // first atom in angle
            if (count(masked_atoms.begin(), masked_atoms.end(), i))
                continue;
            
            for (int j = 0; j < boundary_projected.size(); j++) {
            
                int offset = j + 1;
                a_index = (i - offset + boundary_projected.size()) % static_cast<int>(boundary_projected.size());
                
                if (!count(masked_atoms.begin(), masked_atoms.end(), a_index))
                    break;
            }
            
            // central atom in angle
            b_index = i;
            
            // last atom in angle
            for (int j = 0; j < boundary_projected.size(); j++) {
                
                int offset = j+1;
                c_index = (i + offset + boundary_projected.size()) % static_cast<int>(boundary_projected.size());
            
                if (!count(masked_atoms.begin(), masked_atoms.end(), c_index))
                    break;
            }
        
            // // construct the vectors pointing from the central atom to the other angle atoms
            double a[2] = {boundary_projected[a_index][0], boundary_projected[a_index][1]};
            double b[2] = {boundary_projected[b_index][0], boundary_projected[b_index][1]};
            double c[2] = {boundary_projected[c_index][0], boundary_projected[c_index][1]};

            double bc[2] = {(c[0]-b[0]) , (c[1]-b[1])};
            double ba[2] = {(a[0]-b[0]) , (a[1]-b[1])};

            // get the internal angle of each atom on the boundary
            double bc_cross_ba = -(ba[0]*bc[1]) + (ba[1]*bc[0]);
            double bc_dot_ba = (ba[0]*bc[0]) + (ba[1]*bc[1]);

            if (bc_cross_ba < 0.0)
                interior_thetas[i] = 360.0 - (acos(bc_dot_ba/(sqrt(bc[0]*bc[0]+bc[1]*bc[1]) * sqrt(ba[0]*ba[0]+ba[1]*ba[1])))*(180.0/M_PI));
            else
                interior_thetas[i] = (acos(bc_dot_ba/(sqrt(bc[0]*bc[0]+bc[1]*bc[1]) * sqrt(ba[0]*ba[0]+ba[1]*ba[1])))*(180.0/M_PI));
        }        

        // checking for potential angles:
        // test (1): ear formed if boundary atom is convex (theta < 180 deg)
        // test (2): the ear contains no reflex atoms (use barycentric coodrinates)
        
        std::vector<std::vector<int>> potential_angles;

        for (int i = 0; i < boundary_projected.size(); i++) {
            
            // getting the indexes of the proposed angle, skipping over the masked atoms that have already had angles defined for them
            int a_index, b_index, c_index;

            // first atom in angle
            if (count(masked_atoms.begin(), masked_atoms.end(), i))
                continue;
            
            for (int j = 0; j < boundary_projected.size(); j++) {
            
                int offset = j + 1;
                a_index = (i - offset + boundary_projected.size()) % static_cast<int>(boundary_projected.size());
                
                if (!count(masked_atoms.begin(), masked_atoms.end(), a_index))
                    break;
            }
            
            // central atom in angle
            b_index = i;
            
            // last atom in angle
            for (int j = 0; j < boundary_projected.size(); j++) {
                
                int offset = j+1;
                c_index = (i + offset + boundary_projected.size()) % static_cast<int>(boundary_projected.size());
            
                if (!count(masked_atoms.begin(), masked_atoms.end(), c_index))
                    break;
            }

            // test (1)
            if (interior_thetas[i] >= 180.0)
                continue;
            
            // test(2)
            if (valid_angle_check(a_index, b_index, c_index, boundary_projected, interior_thetas))
                potential_angles.push_back({a_index, b_index, c_index});
        }

        // choose the potential angle which has the minimum central atom interior_theta
        double min_theta = BIG;
        std::vector<int> best_choice = {0,0,0};
        for (int i = 0; i < potential_angles.size(); i++) {

            if (interior_thetas[potential_angles[i][1]] < min_theta) {
                min_theta = interior_thetas[potential_angles[i][1]];
                best_choice = potential_angles[i];  
            }
        }

        // if no best choice was selected then no potential angles were found and remeshing has failed
        if (sqrt(best_choice[0]*best_choice[0] + best_choice[1]*best_choice[1] + best_choice[2]*best_choice[2]) == 0.0) {
          utils::logmesg(lmp, "Remeshing atom: {}, Position: ({}, {}, {})\n", remove_tag, removed_atom_position[0], removed_atom_position[1] , removed_atom_position[2]);
          utils::logmesg(lmp, "Boundary positions for atom {}:\n", remove_tag);
          for (int q = 0; q < (boundary).size(); q++) {
            utils::logmesg(lmp, "[{}] Atom {} position: ({}, {}, {})\n", remove_tag, (boundary)[q],  boundary_coords[q][0] , boundary_coords[q][1] , boundary_coords[q][2]);
          }
          
          utils::logmesg(lmp, "\nProjected boundary positions for atom {}:\n", remove_tag);
          for (int q = 0; q < (boundary_projected).size(); q++) {
            utils::logmesg(lmp, "[{}] Atom {} projected position: ({}, {})\n", remove_tag, (boundary)[q], boundary_projected[q][0], boundary_projected[q][1]);
          }
          error->one(FLERR, "fix rigid/abrade: Failed to remesh atom {}", remove_tag);
        } 

        // remove this central atom from the boundary that is considered for further re-meshing
        masked_atoms.push_back(best_choice[1]);

        // see if it is possible to swap vertices with an existing remeshed angle to prevent the formation of skinny triangles

        // find the longest edge and largest theta in the proposed angle
        double thetas[3];
        triangle_thetas(best_choice[0], best_choice[1], best_choice[2], boundary_projected, thetas);
        min_theta = *std::min_element(thetas, thetas + 3);
        double max_theta = *std::max_element(thetas, thetas + 3);
    
        int longest_edge[2];
        int index_of_largest;

        if (thetas[0] == max_theta) {
            index_of_largest = best_choice[0];
            longest_edge[0] = best_choice[1];
            longest_edge[1] = best_choice[2];
        } else if (thetas[1] == max_theta) {
            index_of_largest = best_choice[1];
            longest_edge[0] = best_choice[2];
            longest_edge[1] = best_choice[0];
        } else {
            index_of_largest = best_choice[2];
            longest_edge[0] = best_choice[0];
            longest_edge[1] = best_choice[1];
        }

        // checking if longest edge exists in any previously defined angles  

        for (int i = 0; i < remeshed_angles.size(); i++) {
            std::vector<int> angle = remeshed_angles[i];
            for (int j = 0; j < 3; j++) {
                if (angle[j] == longest_edge[1]) {
                    if (angle[(j+1)%3] == longest_edge[0]) {
                        
                        // storing the minimum vertex angle between the proposed angle and the existing angle
                        double old_thetas[3];
                        triangle_thetas(angle[0], angle[1], angle[2], boundary_projected, old_thetas);
                        double old_min_theta = std::min(*std::min_element(old_thetas, old_thetas + 3), min_theta);

                        // contructing the quad boundary from the current best choice and the other atom from the previously defined angle
                        
                        int temp_flip_angle[3] = {angle[0], angle[1], angle[2]};
                        int temp_best_choice[3] = {best_choice[0], best_choice[1], best_choice[2]};
                        int quad_corner = temp_flip_angle[(j+2)%3];
                        int temp[3] = {temp_best_choice[0], temp_best_choice[1], temp_best_choice[2]};

                        // order temp_best_choice until the index of the largest is central
                        int temp_0, temp_1, temp_2;
                        while (temp[1] != index_of_largest){
                            temp_0 = temp[0];
                            temp_1 = temp[1];
                            temp_2 = temp[2];
                            temp[1] = temp_0;
                            temp[2] = temp_1;
                            temp[0] = temp_2;
                        }

                        int quad_boundary[4] = {temp[0], temp[1], temp[2], quad_corner};

                        // flipping the diagonal of the quad formed by the two angles
                        for (int k = 0; k < 3; k++) {
                            if (temp_flip_angle[k] == longest_edge[1])
                                temp_flip_angle[k] = index_of_largest;
                            if (temp_best_choice[k] == longest_edge[0])
                                temp_best_choice[k] = quad_corner;
                        }
                        
                        // checking if these alternate angles are valid
                        
                        if (valid_angle_check(temp_flip_angle[0], temp_flip_angle[1], temp_flip_angle[2], boundary_projected, interior_thetas) && valid_angle_check(temp_best_choice[0], temp_best_choice[1], temp_best_choice[2], boundary_projected, interior_thetas)) {
                            
                            
                            // finding the min theta in the alternate angles
                            double new_thetas_flip[3];
                            triangle_thetas(temp_flip_angle[0], temp_flip_angle[1], temp_flip_angle[2], boundary_projected, new_thetas_flip);
                            
                            double new_thetas_best[3];
                            triangle_thetas(temp_best_choice[0], temp_best_choice[1], temp_best_choice[2], boundary_projected, new_thetas_best);
                            
                            double new_min_theta = std::min(*std::min_element(new_thetas_flip, new_thetas_flip + 3), *std::min_element(new_thetas_best, new_thetas_best + 3));
                            
                            // only accepting them if the alternate angles generate less skinny triangles
                            
                            if (old_min_theta < new_min_theta) {
                                remeshed_angles[i][0] = temp_flip_angle[0];
                                remeshed_angles[i][1] = temp_flip_angle[1];
                                remeshed_angles[i][2] = temp_flip_angle[2];
                                best_choice[0] = temp_best_choice[0];
                                best_choice[1] = temp_best_choice[1];
                                best_choice[2] = temp_best_choice[2];
                            }
                        }
                    }
                }
            }
        }
        remeshed_angles.push_back(best_choice);
    }


    // End of remeshing scheme
    for (int i = 0; i < (remeshed_angles).size(); i++) {
      for (int j = 0; j < (remeshed_angles)[i].size(); j++) {}
      new_angles_list[dlist_i].push_back({(boundary)[(remeshed_angles)[i][0]],
                                          (boundary)[(remeshed_angles)[i][1]],
                                          (boundary)[(remeshed_angles)[i][2]]});
    }
  }  

  // Forward comm new_angles_list so they can be defined around central atoms which are locally stored
  commflag = NEW_ANGLES;
  comm->forward_comm(this);

  // create angle around atom m which is a locally owned atom and the central atom in the angle -  modified from fix_bond_create.cpp
  for (int dlist_i = 0; dlist_i < dlist.size(); dlist_i++) {
    for (int a = 0; a < (new_angles_list[dlist_i].size()); a++) {

      // Skip the new angle if the central atom is not locally owned (and not a ghost atom)
      int m = atom->map(new_angles_list[dlist_i][a][1]);
      if ((m < 0) || (m >= nlocal)) continue;

      // if atom is locally owned then it should have an assocaited angle type assigned to it
      if (new_angles_type[dlist_i] == -1) {
        error->one(FLERR, "Remeshed angle type not set for dlist atom {}", dlist[dlist_i]);
      }

      // checking that the other atoms in the angle are stored as ghost atoms
      if (atom->map(new_angles_list[dlist_i][a][0]) < 0 ||
          atom->map(new_angles_list[dlist_i][a][1]) < 0 ||
          atom->map(new_angles_list[dlist_i][a][2]) < 0)
        error->one(FLERR, "Fix Rigid/Abrade re-meshing needs ghost atoms from further away");

      // reading in proc angle information for atom central atom in new angle
      int num_angle = atom->num_angle[m];
      int *angle_type = atom->angle_type[m];
      tagint *angle_atom1 = atom->angle_atom1[m];
      tagint *angle_atom2 = atom->angle_atom2[m];
      tagint *angle_atom3 = atom->angle_atom3[m];

      // check there is space to add an extra angle for the atom
      if (num_angle >= atom->angle_per_atom)
        error->one(FLERR, "Fix Rigid/Abrade induced too many angles per atom during remeshing");
      else {
        
       // define angle about central atom
        angle_type[num_angle] = new_angles_type[dlist_i];

        angle_atom1[num_angle] = new_angles_list[dlist_i][a][0];
        angle_atom2[num_angle] = new_angles_list[dlist_i][a][1];
        angle_atom3[num_angle] = new_angles_list[dlist_i][a][2];
        num_angle++;
      }

      atom->num_angle[m] = num_angle;
    }
  }

  // Locally building the topology
  int i, m, atom1, atom2, atom3;

  nlocal = atom->nlocal;
  int *num_angle = atom->num_angle;
  tagint **angle_atom1 = atom->angle_atom1;
  tagint **angle_atom2 = atom->angle_atom2;
  tagint **angle_atom3 = atom->angle_atom3;
  int **angle_type = atom->angle_type;
  int newton_bond = force->newton_bond;

  int nanglelist = neighbor->nanglelist;
  int **anglelist = neighbor->anglelist;
  int angle_count = 0;
  overflow_anglelist.clear();

  // cycle through each atom coreesponding to each local atom
  for (i = 0; i < nlocal; i++)
    for (m = 0; m < num_angle[i]; m++) {

      atom1 = atom->map(angle_atom1[i][m]);
      atom2 = atom->map(angle_atom2[i][m]);
      atom3 = atom->map(angle_atom3[i][m]);

      // place angle in the existing anglelist if there is space
      if (angle_count < nanglelist) {

        anglelist[angle_count][0] = atom1;
        anglelist[angle_count][1] = atom2;
        anglelist[angle_count][2] = atom3;
        anglelist[angle_count][3] = angle_type[i][m];
      }

      // append angle onto the overflow vector if there is no more space in anglelist
      else {
        overflow_anglelist.push_back({atom1, atom2, atom3, angle_type[i][m]});
      }

      angle_count++;
    }

  remesh_angle_proc_difference = (angle_count - neighbor->nanglelist);
  if (remesh_angle_proc_difference < 0) remesh_overflow_threshold = remesh_angle_proc_difference;

  // The locally rebuilt topology which may spill over into an overflow list

  // The original angles assigned to the dlist atom are not included in the locally rebuilt topology
  // However, the overlflow is required since angles may not be defined on the same proc which deleted them
  // Eg. a proc that hasn't deleted any (which has a full nanglelist) may need to define a new angle if it owns the central atom

  // dlist atoms are not formally removed yet, therefore there is no need to adjust vertexdata[i][], displce[i][] etc. yet

  // now need to cycle through angle lists for the locally rebuild topology:
  //      - dont exceed number of angles if it is less than neighbor->nanglelist
  //      - if overflow exists, cycle through that aswell

  // Recall areas_and_normals() which can in turn recall remesh() if the surface atoms are still too crowded
  areas_and_normals();
}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::compute_forces_and_torques()
{
  int i, ibody;

  // sum over atoms to get force and torque on rigid body

  double **x = atom->x;
  double **f = atom->f;
  int nlocal = atom->nlocal;

  double dx, dy, dz;
  double *xcm, *fcm, *tcm;

  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    fcm = body[ibody].fcm;
    fcm[0] = fcm[1] = fcm[2] = 0.0;
    tcm = body[ibody].torque;
    tcm[0] = tcm[1] = tcm[2] = 0.0;
  }

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    fcm = b->fcm;
    fcm[0] += f[i][0];
    fcm[1] += f[i][1];
    fcm[2] += f[i][2];

    domain->unmap(x[i], xcmimage[i], unwrap[i]);
    xcm = b->xcm;
    dx = unwrap[i][0] - xcm[0];
    dy = unwrap[i][1] - xcm[1];
    dz = unwrap[i][2] - xcm[2];

    tcm = b->torque;
    tcm[0] += dy * f[i][2] - dz * f[i][1];
    tcm[1] += dz * f[i][0] - dx * f[i][2];
    tcm[2] += dx * f[i][1] - dy * f[i][0];
  }

  // reverse communicate fcm, torque of all bodies

  commflag = FORCE_TORQUE;
  comm->reverse_comm(this, 6);

  // include Langevin thermostat forces and torques

  if (langflag) {
    for (ibody = 0; ibody < nlocal_body; ibody++) {
      fcm = body[ibody].fcm;
      fcm[0] += langextra[ibody][0];
      fcm[1] += langextra[ibody][1];
      fcm[2] += langextra[ibody][2];
      tcm = body[ibody].torque;
      tcm[0] += langextra[ibody][3];
      tcm[1] += langextra[ibody][4];
      tcm[2] += langextra[ibody][5];
    }
  }

  // add gravity force to COM of each body

  if (id_gravity) {
    double mass;
    for (ibody = 0; ibody < nlocal_body; ibody++) {
      mass = body[ibody].mass;
      fcm = body[ibody].fcm;
      fcm[0] += gvec[0] * mass;
      fcm[1] += gvec[1] * mass;
      fcm[2] += gvec[2] * mass;
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::final_integrate()
{

  double dtfm;

  // compute forces and torques (after all post_force contributions)
  // if 2d model, enforce2d() on body forces/torques

  if (!earlyflag) compute_forces_and_torques();
  if (domain->dimension == 2) enforce2d();

  // update vcm and angmom, recompute omega, zero wear energy for ghost bodies

  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];

    // update vcm by 1/2 step

    dtfm = dtf / b->mass;
    b->vcm[0] += dtfm * b->fcm[0] * dynamic_flag;
    b->vcm[1] += dtfm * b->fcm[1] * dynamic_flag;
    b->vcm[2] += dtfm * b->fcm[2] * dynamic_flag;

    // update angular momentum by 1/2 step

    b->angmom[0] += dtf * b->torque[0] * dynamic_flag;
    b->angmom[1] += dtf * b->torque[1] * dynamic_flag;
    b->angmom[2] += dtf * b->torque[2] * dynamic_flag;

    MathExtra::angmom_to_omega(b->angmom, b->ex_space, b->ey_space, b->ez_space, b->inertia,
                               b->omega);
  }

  // forward communicate updated info of all bodies

  commflag = FINAL;
  comm->forward_comm(this, 10);

  // set velocity/rotation of atoms in rigid bodies
  // virial is already setup from initial_integrate

  set_v();

  // Integrate the body postitions, stored in displace[i], by their calculated displacement velocities to move atoms within the body
  // This is placed in the final integration step after the centre of mass of each body has been integrated.
  // Thus, atoms are displaced with respect the updated COM position.

  double **x = atom->x;
  int nlocal = atom->nlocal;
  double global_displace_vel[3];
  double body_displace_vel[3];

  int xbox, ybox, zbox;
  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;

  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;

  for (int i = 0; i < nlocal; i++) {

    // Checking that atom i is in a rigid body
    if (atom2body[i] < 0) continue;
      
    // check if the atom has been assigned an abrasion velocity.
    if ((!vertexdata[i][4]) && (!vertexdata[i][5]) && (!vertexdata[i][6])) continue;

    
    global_displace_vel[0] = vertexdata[i][4];
    global_displace_vel[1] = vertexdata[i][5];
    global_displace_vel[2] = vertexdata[i][6];

    // Flagging that the body owning atom i has been abraded and has changed shape
    Body *b = &body[atom2body[i]];
    b->abraded_flag = 1;

    // Convert displacement velocities from global coordinates to body coordinates
    MathExtra::transpose_matvec(b->ex_space, b->ey_space, b->ez_space, global_displace_vel,
                                body_displace_vel);

    // Integrate the position of atom i within the body by its displacement velocity
    displace[i][0] += dtv * body_displace_vel[0];
    displace[i][1] += dtv * body_displace_vel[1];
    displace[i][2] += dtv * body_displace_vel[2];

    // Update the cumulative displacement of each atom
    vertexdata[i][7] += sqrt(((dtv * body_displace_vel[0]) * (dtv * body_displace_vel[0])) +
                             ((dtv * body_displace_vel[1]) * (dtv * body_displace_vel[1])) +
                             ((dtv * body_displace_vel[2]) * (dtv * body_displace_vel[2])));

    // Convert the postiion of atom i from body coordinates to global coordinates
    MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, displace[i], x[i]);

    // This transormation is with respect to (0,0,0) in the global coordinate space, need to translate the position of atom i by the postition of its body's COM
    // Additionally,map back into periodic box via xbox,ybox,zbox
    // same for triclinic, add in box tilt factors as well

    xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
    ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;

    // add center of mass to displacement
    if (triclinic == 0) {
      x[i][0] += b->xcm[0] - xbox * xprd;
      x[i][1] += b->xcm[1] - ybox * yprd;
      x[i][2] += b->xcm[2] - zbox * zprd;
    } else {
      x[i][0] += b->xcm[0] - xbox * xprd - ybox * xy - zbox * xz;
      x[i][1] += b->xcm[1] - ybox * yprd - zbox * yz;
      x[i][2] += b->xcm[2] - zbox * zprd;
    }
  }

  // reverse and forward communication are required to consistently set the abraded_flag for both owned and ghost bodies across all processors

  commflag = ABRADED_FLAG;
  comm->reverse_comm(this, 1);

  commflag = ABRADED_FLAG;
  comm->forward_comm(this, 1);

  // reverse comm cumulative wear energy from ghost bodies
  commflag = WEAR_ENERGY;
  comm->reverse_comm(this, 1);

  // clear for ghost bodies so that they are not doubly counted on the following timesteps
  // also check if any abrasion has occured to any stored bodies
  for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) {
    
    if (body[ibody].abraded_flag) {
      proc_abraded_flag = 1;
      break;
      }
    
    if (ibody < nlocal_body) continue;
    body[ibody].wear_energy = 0.0;
  }

  MPI_Allreduce(&proc_abraded_flag, &global_abraded_flag, 1, MPI_INT, MPI_SUM, world);

  if (global_abraded_flag){

    if (proc_abraded_flag) {
      offset_vertices_outwards();
    }

    // recalculate properties and normals for each abraded body
    resetup_bodies_static();

    // forward comm displacement velocities so that preference can be given to abrading atoms during remeshing
    if (remesh_flag) {
    commflag = DISPLACEMENT_VEL;
    comm->forward_comm(this, 3);
    }
    
    // recalculate areas and normals using the updated body coordinates stored in displace[i] (also check if remeshing is required)
    areas_and_normals();

    if (proc_abraded_flag)
      offset_vertices_inwards();

    // Setting the displacement velocities of all atoms back to 0
    for (int i = 0; i < (nlocal + atom->nghost); i++) {
      vertexdata[i][4] = 0.0;
      vertexdata[i][5] = 0.0;
      vertexdata[i][6] = 0.0;
    }
    
    // setting all abraded_flags for owned and ghost bodies back to 0 in preparation for the following timestep
    for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) body[ibody].abraded_flag = 0;
  }
  
  global_abraded_flag = 0;
  proc_abraded_flag = 0;


  // optionally storing normals in global coords for visualisation
  if (global_normals_flag){
    double bodynormals[3], global_normals[3];  
    for (int i = 0; i < atom->nlocal; i++) {
    
      // Checking that atom i is in a rigid body
    if (atom2body[i] < 0) continue;

    bodynormals[0] = vertexdata[i][0];
    bodynormals[1] = vertexdata[i][1];
    bodynormals[2] = vertexdata[i][2];

    Body *b = &body[atom2body[i]];
    MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, bodynormals, global_normals);

    vertexdata[i][8]  = global_normals[0];
    vertexdata[i][9]  = global_normals[1];
    vertexdata[i][10] = global_normals[2];
  }
}


}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::end_of_step()
{


  // if remeshing has been processed on the current timestep rebuild the neighbor list and formally remove dlist atoms
  if (rebuild_flag) {

    // decrement atom->nangles by global number of angle change
    int all;
    MPI_Allreduce(&remesh_angle_proc_difference, &all, 1, MPI_INT, MPI_SUM, world);
    atom->nangles += all;

    // delete local atoms that have remove_tag and reset nlocal
    int nlocal = atom->nlocal;
    AtomVec *avec = atom->avec;

    int i = 0;
    int removed_count = 0;
    while (i < nlocal) {
      if (count(total_dlist.begin(), total_dlist.end(), atom->tag[i])) {
        removed_count++;
        avec->copy(
            nlocal - 1, i,
            1);    // adjusting vertexdata[i][n], image flags, displace, unwraped since avec->copy(nlocal - 1, i, 1) also invokes fix rigid/abrade's copy_arrays()
        nlocal--;
      } else
        i++;
    }

    int remove_count_all = 0;
    MPI_Allreduce(&removed_count, &remove_count_all, 1, MPI_LMP_BIGINT, MPI_SUM, world);

    if (me ==0 && update->ntimestep == 0 && initial_remesh_flag)
      utils::logmesg(lmp, "fix rigid/abrade: {} atoms remeshed on timestep {}\n", remove_count_all, update->ntimestep);

    // resetting nlocal
    atom->nlocal = nlocal;

    
    // reset atom->natoms and also topology counts
    bigint nblocal = atom->nlocal;
    MPI_Allreduce(&nblocal, &atom->natoms, 1, MPI_LMP_BIGINT, MPI_SUM, world);

    // reset atom->map if it exists
    // set nghost to 0 so old ghosts of deleted atoms won't be mapped
    if (atom->map_style != Atom::MAP_NONE) {
      atom->nghost = 0;
      atom->map_init();
      atom->map_set();
    }


    // Getting a list of fixes so their pre_neighbor() and post_neighbor() functions can also be called
    std::vector<Fix *> fix_list = modify->get_fix_list();
    // rebuilding neighbor lists
    for (auto &fix_i : fix_list) fix_i->pre_exchange();

    if (domain->triclinic) domain->x2lamda(atom->nlocal+atom->nghost);
    
    domain->pbc();
    domain->reset_box();
    comm->setup();
    neighbor->setup_bins();
    comm->exchange();
    comm->borders();
    
    if (domain->triclinic) domain->lamda2x(atom->nlocal+atom->nghost);

    for (auto &fix_i : fix_list) fix_i->pre_neighbor();
    neighbor->build(1);
    for (auto &fix_i : fix_list) fix_i->post_neighbor();

    neighbor->ago = -1;

    // clearing remeshing arrays and resetting flags
    dlist.clear();
    body_dlist.clear();
    
    // Set that the neighbor list was last updates -1 timesteps ago
    // This carries this rebuild over into the next timestep for fixes (such as rigid/wall/region) that dont have a post_neighbor() function but which still check if a remeshing has happened through neighbor->ago == 0
    // If the neighbor list is rebuilt on the next step this is simply set to 0 then, if not it is incremented in neighbor->decide() to 0, thus carrying forward this rebuild into the next timestep
    
    // Need to resetup remeshed bodies to update body coordinates and subsequntly the normals of their surface atoms.
    // reset interia, volume, mass, and most importantly the COM, principal axes, and displace[i] used in areas_and_normals()

    for (int ibody = 0; ibody < (nlocal_body); ibody++) {
      if (count(total_body_dlist.begin(), total_body_dlist.end(), atom->tag[body[ibody].ilocal]))
        body[ibody].abraded_flag = 1;
    } 

    commflag = ABRADED_FLAG;
    comm->forward_comm(this,1);

    for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++)
      if (body[ibody].abraded_flag)
        proc_abraded_flag = 1;

    total_dlist.clear();
    total_body_dlist.clear();
    
    proc_remesh_flag = 0;
    global_remesh_flag = 0;
    rebuild_flag = 0;

    remesh_angle_proc_difference = 0;
    remesh_overflow_threshold = 0;
    overflow_anglelist.clear();

    // Forward comm the bodies so that reset_atom2body can be assigned.
    nghost_body = 0;
    commflag = FULL_BODY;
    comm->forward_comm(this);

    reset_atom2body();

    offset_vertices_outwards();

    resetup_bodies_static();
    
    areas_and_normals();

    offset_vertices_inwards();

    for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++)
      {
        body[ibody].abraded_flag = 0;
      }
    proc_abraded_flag = 0;

    // Reset displacement velocity as a a precaution
    for (int i = nlocal; i < (atom->nlocal + atom->nghost); i++) {
      vertexdata[i][4] = vertexdata[i][5] = vertexdata[i][6] = 0.0; // displacement velocity (x,y,z)
    }

  }

  if (update->ntimestep == output->next_restart)
    restart_file = 1;

}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::initial_integrate_respa(int vflag, int ilevel, int /*iloop*/)
{
  dtv = step_respa[ilevel];
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  dtq = 0.5 * step_respa[ilevel];

  if (ilevel == 0)
    initial_integrate(vflag);
  else
    final_integrate();
}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::final_integrate_respa(int ilevel, int /*iloop*/)
{
  dtf = 0.5 * step_respa[ilevel] * force->ftm2v;
  final_integrate();
}

/* ----------------------------------------------------------------------
   remap xcm of each rigid body back into periodic simulation box
   done during pre_neighbor so will be after call to pbc()
     and after fix_deform::pre_exchange() may have flipped box
   use domain->remap() in case xcm is far away from box
     due to first-time definition of rigid body in setup_bodies_static()
     or due to box flip
   also adjust imagebody = rigid body image flags, due to xcm remap
   then communicate bodies so other procs will know of changes to body xcm
   then adjust xcmimage flags of all atoms in bodies via image_shift()
     for two effects
     (1) change in true image flags due to pbc() call during exchange
     (2) change in imagebody due to xcm remap
   xcmimage flags are always -1,0,-1 so that body can be unwrapped
     around in-box xcm and stay close to simulation box
   if just inferred unwrapped from atom image flags,
     then a body could end up very far away
     when unwrapped by true image flags
   then set_xv() will compute huge displacements every step to reset coords of
     all the body atoms to be back inside the box, ditto for triclinic box flip
     note: so just want to avoid that numeric problem?
------------------------------------------------------------------------- */

void FixRigidAbrade::pre_neighbor()
{
  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    Body *b = &body[ibody];
    domain->remap(b->xcm, b->image);
  }

  nghost_body = 0;
  commflag = FULL_BODY;
  comm->forward_comm(this);
  reset_atom2body();
  //check(4);

  image_shift();
}

/* ----------------------------------------------------------------------
   reset body xcmimage flags of atoms in bodies
   xcmimage flags are relative to xcm so that body can be unwrapped
   xcmimage = true image flag - imagebody flag
------------------------------------------------------------------------- */

void FixRigidAbrade::image_shift()
{
  imageint tdim, bdim, xdim[3];

  imageint *image = atom->image;
  int nlocal = atom->nlocal;

  for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++){
    atom->image[body[ibody].ilocal] = body[ibody].image;
  }

  for (int i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    tdim = image[i] & IMGMASK;
    bdim = b->image & IMGMASK;
    xdim[0] = IMGMAX + tdim - bdim;
    tdim = (image[i] >> IMGBITS) & IMGMASK;
    bdim = (b->image >> IMGBITS) & IMGMASK;
    xdim[1] = IMGMAX + tdim - bdim;
    tdim = image[i] >> IMG2BITS;
    bdim = b->image >> IMG2BITS;
    xdim[2] = IMGMAX + tdim - bdim;

    xcmimage[i] = (xdim[2] << IMG2BITS) | (xdim[1] << IMGBITS) | xdim[0];
  }
}

/* ----------------------------------------------------------------------
   count # of DOF removed by rigid bodies for atoms in igroup
   return total count of DOF
------------------------------------------------------------------------- */

bigint FixRigidAbrade::dof(int tgroup)
{
  int i, j;

  // cannot count DOF correctly unless setup_bodies_static() has been called

  if (!setupflag) {
    if (comm->me == 0)
      error->warning(FLERR,
                     "Cannot count rigid body degrees-of-freedom "
                     "before bodies are fully initialized");
    return 0;
  }

  int tgroupbit = group->bitmask[tgroup];

  // counts = 3 values per rigid body I own
  // 0 = # of point particles in rigid body and in temperature group
  // 1 = # of finite-size particles in rigid body and in temperature group
  // 2 = # of particles in rigid body, disregarding temperature group

  memory->create(counts, nlocal_body + nghost_body, 3, "rigid/abrade:counts");
  for (i = 0; i < nlocal_body + nghost_body; i++) counts[i][0] = counts[i][1] = counts[i][2] = 0;

  // tally counts from my owned atoms
  // 0 = # of point particles in rigid body and in temperature group
  // 1 = # of finite-size particles in rigid body and in temperature group
  // 2 = # of particles in rigid body, disregarding temperature group

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    j = atom2body[i];
    counts[j][2]++;
    if (mask[i] & tgroupbit) {
      if (extended && (eflags[i]))
        counts[j][1]++;
      else
        counts[j][0]++;
    }
  }

  commflag = DOF;
  comm->reverse_comm(this, 3);

  // nall = count0 = # of point particles in each rigid body
  // mall = count1 = # of finite-size particles in each rigid body
  // warn if nall+mall != nrigid for any body included in temperature group

  int flag = 0;
  for (int ibody = 0; ibody < nlocal_body; ibody++) {
    if (counts[ibody][0] + counts[ibody][1] > 0 &&
        counts[ibody][0] + counts[ibody][1] != counts[ibody][2])
      flag = 1;
  }
  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);
  if (flagall && comm->me == 0)
    error->warning(FLERR, "Computing temperature of portions of rigid bodies");

  // remove appropriate DOFs for each rigid body wholly in temperature group
  // N = # of point particles in body
  // M = # of finite-size particles in body
  // 3d body has 3N + 6M dof to start with
  // 2d body has 2N + 3M dof to start with
  // 3d point-particle body with all non-zero I should have 6 dof, remove 3N-6
  // 3d point-particle body (linear) with a 0 I should have 5 dof, remove 3N-5
  // 2d point-particle body should have 3 dof, remove 2N-3
  // 3d body with any finite-size M should have 6 dof, remove (3N+6M) - 6
  // 2d body with any finite-size M should have 3 dof, remove (2N+3M) - 3

  double *inertia;

  bigint n = 0;
  nlinear = 0;
  if (domain->dimension == 3) {
    for (int ibody = 0; ibody < nlocal_body; ibody++) {
      if (counts[ibody][0] + counts[ibody][1] == counts[ibody][2]) {
        n += 3 * counts[ibody][0] + 6 * counts[ibody][1] - 6;
        inertia = body[ibody].inertia;
        if (inertia[0] == 0.0 || inertia[1] == 0.0 || inertia[2] == 0.0) {
          n++;
          nlinear++;
        }
      }
    }
  } else if (domain->dimension == 2) {
    for (int ibody = 0; ibody < nlocal_body; ibody++)
      if (counts[ibody][0] + counts[ibody][1] == counts[ibody][2])
        n += 2 * counts[ibody][0] + 3 * counts[ibody][1] - 3;
  }

  memory->destroy(counts);

  bigint nall;
  MPI_Allreduce(&n, &nall, 1, MPI_LMP_BIGINT, MPI_SUM, world);
  return nall;
}

/* ----------------------------------------------------------------------
   adjust xcm of each rigid body due to box deformation
   called by various fixes that change box size/shape
   flag = 0/1 means map from box to lamda coords or vice versa
------------------------------------------------------------------------- */

void FixRigidAbrade::deform(int flag)
{
  if (flag == 0)
    for (int ibody = 0; ibody < nlocal_body; ibody++)
      domain->x2lamda(body[ibody].xcm, body[ibody].xcm);
  else
    for (int ibody = 0; ibody < nlocal_body; ibody++)
      domain->lamda2x(body[ibody].xcm, body[ibody].xcm);
}

/* ----------------------------------------------------------------------
   set space-frame coords and velocity of each atom in each rigid body
   x = Q displace + Xcm, mapped back to periodic box
   v = Vcm + (W cross (x - Xcm))
------------------------------------------------------------------------- */

void FixRigidAbrade::set_xv()
{
  int xbox, ybox, zbox;
  double x0, x1, x2, v0, v1, v2, fc0, fc1, fc2, massone;
  double ione[3], exone[3], eyone[3], ezone[3], vr[6], p[3][3];

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // set x and v of each atom

  for (int i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
    ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
    zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;

    // save old positions and velocities for virial

    if (evflag) {
      if (triclinic == 0) {
        x0 = x[i][0] + xbox * xprd;
        x1 = x[i][1] + ybox * yprd;
        x2 = x[i][2] + zbox * zprd;
      } else {
        x0 = x[i][0] + xbox * xprd + ybox * xy + zbox * xz;
        x1 = x[i][1] + ybox * yprd + zbox * yz;
        x2 = x[i][2] + zbox * zprd;
      }
      v0 = v[i][0];
      v1 = v[i][1];
      v2 = v[i][2];
    }

    // x = displacement from center-of-mass, based on body orientation
    // v = vcm + omega around center-of-mass

    MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, displace[i], x[i]);

    v[i][0] = b->omega[1] * x[i][2] - b->omega[2] * x[i][1] + b->vcm[0];
    v[i][1] = b->omega[2] * x[i][0] - b->omega[0] * x[i][2] + b->vcm[1];
    v[i][2] = b->omega[0] * x[i][1] - b->omega[1] * x[i][0] + b->vcm[2];

    // add center of mass to displacement
    // map back into periodic box via xbox,ybox,zbox
    // for triclinic, add in box tilt factors as well

    if (triclinic == 0) {
      x[i][0] += b->xcm[0] - xbox * xprd;
      x[i][1] += b->xcm[1] - ybox * yprd;
      x[i][2] += b->xcm[2] - zbox * zprd;
    } else {
      x[i][0] += b->xcm[0] - xbox * xprd - ybox * xy - zbox * xz;
      x[i][1] += b->xcm[1] - ybox * yprd - zbox * yz;
      x[i][2] += b->xcm[2] - zbox * zprd;
    }

    // virial = unwrapped coords dotted into body constraint force
    // body constraint force = implied force due to v change minus f external
    // assume f does not include forces internal to body
    // 1/2 factor b/c final_integrate contributes other half
    // assume per-atom contribution is due to constraint force on that atom

    if (evflag) {
      if (rmass)
        massone = rmass[i];
      else
        massone = mass[type[i]];
      fc0 = massone * (v[i][0] - v0) / dtf - f[i][0];
      fc1 = massone * (v[i][1] - v1) / dtf - f[i][1];
      fc2 = massone * (v[i][2] - v2) / dtf - f[i][2];

      vr[0] = 0.5 * x0 * fc0;
      vr[1] = 0.5 * x1 * fc1;
      vr[2] = 0.5 * x2 * fc2;
      vr[3] = 0.5 * x0 * fc1;
      vr[4] = 0.5 * x0 * fc2;
      vr[5] = 0.5 * x1 * fc2;

      double rlist[1][3] = {{x0, x1, x2}};
      double flist[1][3] = {{0.5 * fc0, 0.5 * fc1, 0.5 * fc2}};
      v_tally(1, &i, 1.0, vr, rlist, flist, b->xgc);
    }
  }

  // update the position of geometric center
  for (int ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    Body *b = &body[ibody];
    MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, b->xgc_body, b->xgc);
    b->xgc[0] += b->xcm[0];
    b->xgc[1] += b->xcm[1];
    b->xgc[2] += b->xcm[2];
  }
}

/* ----------------------------------------------------------------------
   set space-frame velocity of each atom in a rigid body
   v = Vcm + (W cross (x - Xcm))
------------------------------------------------------------------------- */

void FixRigidAbrade::set_v()
{
  int xbox, ybox, zbox;
  double x0, x1, x2, v0, v1, v2, fc0, fc1, fc2, massone;
  double ione[3], exone[3], eyone[3], ezone[3], delta[3], vr[6];

  double xprd = domain->xprd;
  double yprd = domain->yprd;
  double zprd = domain->zprd;
  double xy = domain->xy;
  double xz = domain->xz;
  double yz = domain->yz;

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  // set v of each atom

  for (int i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    Body *b = &body[atom2body[i]];

    MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, displace[i], delta);

    // save old velocities for virial

    if (evflag) {
      v0 = v[i][0];
      v1 = v[i][1];
      v2 = v[i][2];
    }

    v[i][0] = b->omega[1] * delta[2] - b->omega[2] * delta[1] + b->vcm[0];
    v[i][1] = b->omega[2] * delta[0] - b->omega[0] * delta[2] + b->vcm[1];
    v[i][2] = b->omega[0] * delta[1] - b->omega[1] * delta[0] + b->vcm[2];

    // virial = unwrapped coords dotted into body constraint force
    // body constraint force = implied force due to v change minus f external
    // assume f does not include forces internal to body
    // 1/2 factor b/c initial_integrate contributes other half
    // assume per-atom contribution is due to constraint force on that atom

    if (evflag) {
      if (rmass)
        massone = rmass[i];
      else
        massone = mass[type[i]];
      fc0 = massone * (v[i][0] - v0) / dtf - f[i][0];
      fc1 = massone * (v[i][1] - v1) / dtf - f[i][1];
      fc2 = massone * (v[i][2] - v2) / dtf - f[i][2];

      xbox = (xcmimage[i] & IMGMASK) - IMGMAX;
      ybox = (xcmimage[i] >> IMGBITS & IMGMASK) - IMGMAX;
      zbox = (xcmimage[i] >> IMG2BITS) - IMGMAX;

      if (triclinic == 0) {
        x0 = x[i][0] + xbox * xprd;
        x1 = x[i][1] + ybox * yprd;
        x2 = x[i][2] + zbox * zprd;
      } else {
        x0 = x[i][0] + xbox * xprd + ybox * xy + zbox * xz;
        x1 = x[i][1] + ybox * yprd + zbox * yz;
        x2 = x[i][2] + zbox * zprd;
      }

      vr[0] = 0.5 * x0 * fc0;
      vr[1] = 0.5 * x1 * fc1;
      vr[2] = 0.5 * x2 * fc2;
      vr[3] = 0.5 * x0 * fc1;
      vr[4] = 0.5 * x0 * fc2;
      vr[5] = 0.5 * x1 * fc2;

      double rlist[1][3] = {{x0, x1, x2}};
      double flist[1][3] = {{0.5 * fc0, 0.5 * fc1, 0.5 * fc2}};
      v_tally(1, &i, 1.0, vr, rlist, flist, b->xgc);
    }
  }
}

/* ----------------------------------------------------------------------
   one-time identification of which atoms are in which rigid bodies
   set bodytag for all owned atoms
------------------------------------------------------------------------- */

void FixRigidAbrade::create_bodies(tagint *bodyID)
{
  int i, m;

  // allocate buffer for input to rendezvous comm
  // ncount = # of my atoms in bodies

  int *mask = atom->mask;
  int nlocal = atom->nlocal;

  int ncount = 0;
  for (i = 0; i < nlocal; i++)
    if (mask[i] & groupbit) ncount++;

  int *proclist;
  memory->create(proclist, ncount, "rigid/abrade:proclist");
  auto *inbuf = (InRvous *) memory->smalloc(ncount*sizeof(InRvous),"rigid/small:inbuf");

  // setup buf to pass to rendezvous comm
  // one BodyMsg datum for each constituent atom
  // datum = me, local index of atom, atomID, bodyID, unwrapped coords
  // owning proc for each datum = random hash of bodyID

  double **x = atom->x;
  tagint *tag = atom->tag;
  imageint *image = atom->image;

  m = 0;
  for (i = 0; i < nlocal; i++) {
    if (!(mask[i] & groupbit)) continue;
    proclist[m] = std::hash<tagint>{}(bodyID[i]) % nprocs;
    inbuf[m].me = me;
    inbuf[m].ilocal = i;
    inbuf[m].atomID = tag[i];
    inbuf[m].bodyID = bodyID[i];
    domain->unmap(x[i], image[i], inbuf[m].x);
    m++;
  }

  // perform rendezvous operation
  // each proc owns random subset of bodies
  // receives all atoms in those bodies
  // func = compute bbox of each body, find atom closest to geometric center

  char *buf;
  int nreturn = comm->rendezvous(RVOUS, ncount, (char *) inbuf, sizeof(InRvous), 0, proclist,
                                 rendezvous_body, 0, buf, sizeof(OutRvous), (void *) this);
  auto *outbuf = (OutRvous *) buf;

  memory->destroy(proclist);
  memory->sfree(inbuf);

  // set bodytag of all owned atoms based on outbuf info for constituent atoms

  for (i = 0; i < nlocal; i++)
    if (!(mask[i] & groupbit)) bodytag[i] = 0;

  for (m = 0; m < nreturn; m++) bodytag[outbuf[m].ilocal] = outbuf[m].atomID;

  memory->sfree(outbuf);

  // maxextent = max of rsqfar across all procs
  // if defined, include molecule->maxextent

  MPI_Allreduce(&rsqfar, &maxextent, 1, MPI_DOUBLE, MPI_MAX, world);
  maxextent = sqrt(maxextent);
  if (onemols) {
    for (i = 0; i < nmol; i++) maxextent = MAX(maxextent, onemols[i]->maxextent);
  }
}

/* ----------------------------------------------------------------------
   process rigid bodies assigned to me
   buf = list of N BodyMsg datums
------------------------------------------------------------------------- */

int FixRigidAbrade::rendezvous_body(int n, char *inbuf, int &rflag, int *&proclist, char *&outbuf,
                                    void *ptr)
{
  int i, m;
  double delx, dely, delz, rsq;
  int *iclose;
  tagint *idclose;
  double *x, *xown, *rsqclose;
  double **bbox, **ctr;

  auto *frsptr = (FixRigidAbrade *) ptr;
  Memory *memory = frsptr->memory;
  Error *error = frsptr->error;
  MPI_Comm world = frsptr->world;

  // setup hash
  // use STL unordered_map instead of atom->map
  //   b/c know nothing about body ID values specified by user
  // ncount = number of bodies assigned to me
  // key = body ID
  // value = index into Ncount-length data structure

  auto *in = (InRvous *) inbuf;
  std::unordered_map<tagint,int> hash;
  tagint id;

  int ncount = 0;
  for (i = 0; i < n; i++) {
    id = in[i].bodyID;
    if (hash.find(id) == hash.end()) hash[id] = ncount++;
  }

  // bbox = bounding box of each rigid body

  memory->create(bbox, ncount, 6, "rigid/abrade:bbox");

  for (m = 0; m < ncount; m++) {
    bbox[m][0] = bbox[m][2] = bbox[m][4] = BIG;
    bbox[m][1] = bbox[m][3] = bbox[m][5] = -BIG;
  }

  for (i = 0; i < n; i++) {
    m = hash.find(in[i].bodyID)->second;
    x = in[i].x;
    bbox[m][0] = MIN(bbox[m][0], x[0]);
    bbox[m][1] = MAX(bbox[m][1], x[0]);
    bbox[m][2] = MIN(bbox[m][2], x[1]);
    bbox[m][3] = MAX(bbox[m][3], x[1]);
    bbox[m][4] = MIN(bbox[m][4], x[2]);
    bbox[m][5] = MAX(bbox[m][5], x[2]);
  }

  // check if any bbox is size 0.0, meaning rigid body is a single particle

  int flag = 0;
  for (m = 0; m < ncount; m++)
    if (bbox[m][0] == bbox[m][1] && bbox[m][2] == bbox[m][3] && bbox[m][4] == bbox[m][5]) flag = 1;
  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_SUM, world);    // sync here?
  if (flagall) error->all(FLERR, "One or more rigid bodies are a single particle");

  // ctr = geometric center pt of each rigid body

  memory->create(ctr, ncount, 3, "rigid/abrade:bbox");

  for (m = 0; m < ncount; m++) {
    ctr[m][0] = 0.5 * (bbox[m][0] + bbox[m][1]);
    ctr[m][1] = 0.5 * (bbox[m][2] + bbox[m][3]);
    ctr[m][2] = 0.5 * (bbox[m][4] + bbox[m][5]);
  }

  // idclose = atomID closest to center point of each body

  memory->create(idclose, ncount, "rigid/abrade:idclose");
  memory->create(iclose, ncount, "rigid/abrade:iclose");
  memory->create(rsqclose, ncount, "rigid/abrade:rsqclose");
  for (m = 0; m < ncount; m++) rsqclose[m] = BIG;

  for (i = 0; i < n; i++) {
    m = hash.find(in[i].bodyID)->second;
    x = in[i].x;
    delx = x[0] - ctr[m][0];
    dely = x[1] - ctr[m][1];
    delz = x[2] - ctr[m][2];
    rsq = delx * delx + dely * dely + delz * delz;
    if (rsq <= rsqclose[m]) {
      if (rsq == rsqclose[m] && in[i].atomID > idclose[m]) continue;
      iclose[m] = i;
      idclose[m] = in[i].atomID;
      rsqclose[m] = rsq;
    }
  }

  // compute rsqfar for all bodies I own
  // set rsqfar back in caller

  double rsqfar = 0.0;

  for (i = 0; i < n; i++) {
    m = hash.find(in[i].bodyID)->second;
    xown = in[iclose[m]].x;
    x = in[i].x;
    delx = x[0] - xown[0];
    dely = x[1] - xown[1];
    delz = x[2] - xown[2];
    rsq = delx * delx + dely * dely + delz * delz;
    rsqfar = MAX(rsqfar, rsq);
  }

  frsptr->rsqfar = rsqfar;

  // pass list of OutRvous datums back to comm->rendezvous

  int nout = n;
  memory->create(proclist, nout, "rigid/abrade:proclist");
  auto *out = (OutRvous *) memory->smalloc(nout*sizeof(OutRvous),"rigid/small:out");

  for (i = 0; i < nout; i++) {
    proclist[i] = in[i].me;
    out[i].ilocal = in[i].ilocal;
    m = hash.find(in[i].bodyID)->second;
    out[i].atomID = idclose[m];
  }

  outbuf = (char *) out;

  // clean up
  // Comm::rendezvous will delete proclist and out (outbuf)

  memory->destroy(bbox);
  memory->destroy(ctr);
  memory->destroy(idclose);
  memory->destroy(iclose);
  memory->destroy(rsqclose);

  // flag = 2: new outbuf

  rflag = 2;
  return nout;
}

/* ----------------------------------------------------------------------
  Pushing atoms outwards from their COM following a read in from a restart 
  so that they can be correctly setup in setup_bodies_static()
------------------------------------------------------------------------- */

void FixRigidAbrade::pre_setup_bodies_static() {
  
  if (!offset_flag) return;

  int nlocal = atom->nlocal;
  double **x = atom->x;
  double *radius = atom->radius;
  double len_i;
  double dx[3];
  
  // Communicate bodies so ghost bodies can be accessed
  nghost_body = 0;
  commflag = FULL_BODY;
  comm->forward_comm(this);
  reset_atom2body();
  
  // marking all bodies as having abraded so required information is communicated
  for (int ibody = 0; ibody < (nlocal_body + nghost_body); ibody++) 
    body[ibody].abraded_flag = 1;
  proc_abraded_flag = 1;

// Cycling through the local atoms and storing their unwrapped coordinates.
for (int i = 0; i < atom->nlocal; i++) {
  if (atom2body[i] < 0) continue;
  domain->unmap(x[i], atom->image[i], unwrap[i]);  
}

// communicate unwrapped position of owned atoms to ghost atoms
commflag = UNWRAP;
comm->forward_comm(this, 3);
  
// offsetting atoms about their body's owning atom which is placed at the COM
  for (int i = 0; i < atom->nlocal; i++) {
    // passing over atoms which own bodies since their position does not need to be offset
    if (bodytag[i] == atom->tag[i]) continue;
    if (atom2body[i] < 0) continue;
  
    dx[0] = unwrap[i][0] - unwrap[atom->map(bodytag[i])][0];
    dx[1] = unwrap[i][1] - unwrap[atom->map(bodytag[i])][1];
    dx[2] = unwrap[i][2] - unwrap[atom->map(bodytag[i])][2];

    // offsetting atoms outwards towards the COM by the atoms radius  
    len_i = MathExtra::len3(dx);
    x[i][0] += radius[i] * (dx[0]/len_i);
    x[i][1] += radius[i] * (dx[1]/len_i);
    x[i][2] += radius[i] * (dx[2]/len_i);
  }

  // Mapping atom positions back to the simulation cell ready for setup_bodies_static()
  for (int i = 0; i < (atom->nlocal); i++)
    domain->remap(atom->x[i], atom->image[i]); 
}

/* ----------------------------------------------------------------------
   one-time initialization of rigid body attributes
   sets extended flags, masstotal, center-of-mass (Note: support for non-spherical rotational extended atoms has been removed for fix rigid/abrade)
   sets Cartesian and diagonalized inertia tensor
   sets body image flags
   Rigid body properties are set by contructing tetrahedra about facets (defined by the angles) with reference to F. Tonon, Explicit Exact Formulas for the 3-D Tetrahedron Inertia Tensor in Terms of its Vertex Coordinates, J. Math. Stat. 1 (2004). doi:10.3844/jmssp.2005.8.11
------------------------------------------------------------------------- */

void FixRigidAbrade::setup_bodies_static()
{

  int i, ibody;

  extended = 0;

  AtomVecEllipsoid::Bonus *ebonus;
  if (avec_ellipsoid) ebonus = avec_ellipsoid->bonus;
  AtomVecLine::Bonus *lbonus;
  if (avec_line) lbonus = avec_line->bonus;
  AtomVecTri::Bonus *tbonus;
  if (avec_tri) tbonus = avec_tri->bonus;
  double **mu = atom->mu;
  double *radius = atom->radius;
  double *rmass = atom->rmass;
  double *mass = atom->mass;
  int *ellipsoid = atom->ellipsoid;
  int *line = atom->line;
  int *tri = atom->tri;
  int *type = atom->type;
  int nlocal = atom->nlocal;

  if (atom->radius_flag) {
    int flag = 0;
    for (i = 0; i < nlocal; i++) {
      if (bodytag[i] == 0) continue;
      if (radius && radius[i] > 0.0) flag = 1;

      if ((ellipsoid && ellipsoid[i] >= 0) || (line && line[i] >= 0)  || (tri && tri[i] >= 0)  || (mu && mu[i][3] > 0.0))
        error->all(FLERR, "fix rigid/abrade requires surface atoms to be arotational finite-size spheres");
    }

    MPI_Allreduce(&flag, &extended, 1, MPI_INT, MPI_MAX, world);
  }

  if (!extended)
     error->all(FLERR, "fix rigid/abrade requires surface atoms to be finite-size spheres");
     
  // extended = 1 if using molecule template with finite-size particles
  // require all molecules in template to have consistent radiusflag

  if (onemols) {
    int radiusflag = onemols[0]->radiusflag;
    for (i = 1; i < nmol; i++) {
      if (onemols[i]->radiusflag != radiusflag)
        error->all(FLERR,
                   "Inconsistent use of finite-size particles "
                   "by molecule template molecules");
    }
    if (radiusflag) extended = 1;
  }

  if (extended) {
   
    grow_arrays(atom->nmax);
    
    for (i = 0; i < nlocal; i++) {
      eflags[i] = 0;
      if (bodytag[i] == 0) continue;
       eflags[i] |= SPHERE;
    }
  }

  // set body xcmimage flags = true image flags

  imageint *image = atom->image;
  for (i = 0; i < nlocal; i++)
    if (bodytag[i] >= 0)
      xcmimage[i] = image[i];
    else
      xcmimage[i] = 0;

  // acquire ghost bodies via forward comm
  // set atom2body for ghost atoms via forward comm
  // set atom2body for other owned atoms via reset_atom2body()

  nghost_body = 0;
  commflag = FULL_BODY;
  comm->forward_comm(this);
  reset_atom2body();

  // compute mass & center-of-mass of each rigid body
  double **x = atom->x;

  double *xcm;
  double *xgc;

  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {

    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;
    xcm[0] = xcm[1] = xcm[2] = 0.0;
    xgc[0] = xgc[1] = xgc[2] = 0.0;
    body[ibody].mass = 0.0;
    body[ibody].volume = 0.0;
    body[ibody].abraded_volume = 0.0;
    body[ibody].wear_energy = 0.0;
    body[ibody].surface_area = 0.0;
    body[ibody].min_area_atom = 0.0;
    body[ibody].min_area_atom_tag = 0;
    body[ibody].density = density;
    body[ibody].natoms = 0;

    // Initally setting all bodies to have been abraded at t = 0 so that they are correctly processed during setup_bodies_static
    body[ibody].abraded_flag = 1;
    body[ibody].remesh_atom = 0;
  }
  proc_abraded_flag = 1;
  
  double massone;

  // Cycling through the local atoms and storing their unwrapped coordinates.
  // Additionally, set increment the number of atoms in their respective body
  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) continue;
    // Calculated unwrapped coords for all atoms in bodies
    domain->unmap(x[i], xcmimage[i], unwrap[i]);

    Body *b = &body[atom2body[i]];
    b->natoms++;
  }

  int nanglelist = neighbor->nanglelist;
  int **anglelist = neighbor->anglelist;

  int i1, i2, i3;

  // communicate unwrapped position of owned atoms to ghost atoms
  commflag = UNWRAP;
  comm->forward_comm(this, 3);

  // Calculating body volume, mass and COM from constituent tetrahedra
  double inverse_24 = 1.0/24.0;
  double inverse_6 = 1.0/6.0;

  for (int n = 0; n < nanglelist; n++) {
    if (atom2body[anglelist[n][0]] < 0) continue;

    Body *b = &body[atom2body[anglelist[n][0]]];

    // Storing the three atoms in each angle
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];

    xcm = b->xcm;
    xgc = b->xgc;

    b->volume += ((((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
                   ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
                  ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])) * inverse_6;

    xcm[0] +=
        ((((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
          ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
         (((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
          unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]))) * inverse_24;
    xcm[1] +=
        ((((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][2] - unwrap[i1][2])) -
          ((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][2] - unwrap[i1][2]))) *
         (((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
          unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]))) * inverse_24;
    xcm[2] +=
        ((((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][1] - unwrap[i1][1])) -
          ((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][1] - unwrap[i1][1]))) *
         (((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
          unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]))) * inverse_24;
    xgc[0] +=
        ((((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
          ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
         (((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
          unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]))) * inverse_24;
    xgc[1] +=
        ((((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][2] - unwrap[i1][2])) -
          ((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][2] - unwrap[i1][2]))) *
         (((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
          unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]))) * inverse_24;
    xgc[2] +=
        ((((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][1] - unwrap[i1][1])) -
          ((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][1] - unwrap[i1][1]))) *
         (((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
          unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]))) * inverse_24;
  }

  // reverse communicate xcm, mass of all bodies
  commflag = XCM_MASS;
  comm->reverse_comm(this, 9);

  double inverse_volume;

  for (ibody = 0; ibody < nlocal_body; ibody++) {

    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;

    // Setting each bodies' COM
    inverse_volume = 1.0 /  body[ibody].volume;
    xcm[0] *= inverse_volume;
    xcm[1] *= inverse_volume;
    xcm[2] *= inverse_volume;
    xgc[0] *= inverse_volume;
    xgc[1] *= inverse_volume;
    xgc[2] *= inverse_volume;

    // Positioning the owning atom at the COM
    unwrap[body[ibody].ilocal][0] = body[ibody].xcm[0];
    unwrap[body[ibody].ilocal][1] = body[ibody].xcm[1];
    unwrap[body[ibody].ilocal][2] = body[ibody].xcm[2];
    x[body[ibody].ilocal][0] = body[ibody].xcm[0];
    x[body[ibody].ilocal][1] = body[ibody].xcm[1];
    x[body[ibody].ilocal][2] = body[ibody].xcm[2];

    // Setting the mass of each body
    body[ibody].mass = body[ibody].volume * density;
    body[ibody].initial_volume = body[ibody].volume;
  }
  
  double *vcm, *angmom;

  for (ibody = 0; ibody < nlocal_body; ibody++) {
    vcm = body[ibody].vcm;
    vcm[0] = vcm[1] = vcm[2] = 0.0;
    angmom = body[ibody].angmom;
    angmom[0] = angmom[1] = angmom[2] = 0.0;
  }

  // set rigid body image flags to default values

  for (ibody = 0; ibody < nlocal_body; ibody++)
    body[ibody].image = ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;

  // remap the xcm of each body back into simulation box
  //   and reset body and atom xcmimage flags via pre_neighbor()

  pre_neighbor();

  // recalculating unwrapped coordinates of all atoms in bodies since have reset xcmimage flags
  for (i = 0; i < nlocal; i++) {

    if (atom2body[i] < 0) continue;

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[atom2body[i]].abraded_flag) continue;
    
    domain->unmap(x[i], xcmimage[i], unwrap[i]);
  
  }


  commflag = UNWRAP;
  comm->forward_comm(this, 3);

  // compute 6 moments of inertia of each body in Cartesian reference frame
  // dx,dy,dz = coords relative to center-of-mass
  // symmetric 3x3 inertia tensor stored in Voigt notation as 6-vector

  // also place owning atom at COM following the mapping of XCM in pre_neighbor

  memory->create(itensor, nlocal_body + nghost_body, 6, "rigid/abrade:itensor");
  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {

    for (i = 0; i < 6; i++) itensor[ibody][i] = 0.0;

    // Positioning the owning atom at the COM
    x[body[ibody].ilocal][0] = body[ibody].xcm[0];
    x[body[ibody].ilocal][1] = body[ibody].xcm[1];
    x[body[ibody].ilocal][2] = body[ibody].xcm[2];

        // Positioning the owning atom at the COM
    unwrap[body[ibody].ilocal][0] = body[ibody].xcm[0];
    unwrap[body[ibody].ilocal][1] = body[ibody].xcm[1];
    unwrap[body[ibody].ilocal][2] = body[ibody].xcm[2];
  }

  double dx, dy, dz;
  double *inertia;

  for (int n = 0; n < nanglelist; n++) {
    if (atom2body[anglelist[n][0]] < 0) continue;
    Body *b = &body[atom2body[anglelist[n][0]]];

    // Storing the three atoms in each angle
    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];

    inertia = itensor[atom2body[anglelist[n][0]]];
    inertia[0] += (((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
                   ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
        (unwrap[i1][0] * (unwrap[i1][0] * unwrap[i1][0]) +
         unwrap[i2][0] *
             ((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
         unwrap[i3][0] *
             (((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
              unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])));
    inertia[1] += (((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][2] - unwrap[i1][2])) -
                   ((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][2] - unwrap[i1][2]))) *
        (unwrap[i1][1] * (unwrap[i1][1] * unwrap[i1][1]) +
         unwrap[i2][1] *
             ((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
         unwrap[i3][1] *
             (((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
              unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1])));
    inertia[2] += (((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][1] - unwrap[i1][1])) -
                   ((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][1] - unwrap[i1][1]))) *
        (unwrap[i1][2] * (unwrap[i1][2] * unwrap[i1][2]) +
         unwrap[i2][2] *
             ((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
         unwrap[i3][2] *
             (((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
              unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2])));
    inertia[3] += (((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
                   ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
        (unwrap[i1][1] *
             ((((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
               unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])) +
              unwrap[i1][0] * (((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]) + unwrap[i1][0])) +
         unwrap[i2][1] *
             ((((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
               unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])) +
              unwrap[i2][0] * (((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]) + unwrap[i2][0])) +
         unwrap[i3][1] *
             ((((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
               unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])) +
              unwrap[i3][0] * (((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]) + unwrap[i3][0])));
    inertia[4] += (((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][2] - unwrap[i1][2])) -
                   ((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][2] - unwrap[i1][2]))) *
        (unwrap[i1][2] *
             ((((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
               unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1])) +
              unwrap[i1][1] * (((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]) + unwrap[i1][1])) +
         unwrap[i2][2] *
             ((((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
               unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1])) +
              unwrap[i2][1] * (((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]) + unwrap[i2][1])) +
         unwrap[i3][2] *
             ((((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
               unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1])) +
              unwrap[i3][1] * (((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]) + unwrap[i3][1])));
    inertia[5] += (((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][1] - unwrap[i1][1])) -
                   ((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][1] - unwrap[i1][1]))) *
        (unwrap[i1][0] *
             ((((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
               unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2])) +
              unwrap[i1][2] * (((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]) + unwrap[i1][2])) +
         unwrap[i2][0] *
             ((((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
               unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2])) +
              unwrap[i2][2] * (((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]) + unwrap[i2][2])) +
         unwrap[i3][0] *
             ((((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
               unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2])) +
              unwrap[i3][2] * (((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]) + unwrap[i3][2])));
  }

  // reverse communicate inertia tensor of all bodies

  commflag = ITENSOR;
  comm->reverse_comm(this, 6);


  // diagonalize inertia tensor for each body via Jacobi rotations
  // inertia = 3 eigenvalues = principal moments of inertia
  // evectors and exzy_space = 3 evectors = principal axes of rigid body

  int ierror;
  double cross[3];
  double tensor[3][3], evectors[3][3];
  double *ex, *ey, *ez;

  double inverse_60 = 1.0 / 60.0;
  double inverse_120 = 1.0 / 120.0;

  for (ibody = 0; ibody < nlocal_body; ibody++) {

    tensor[0][0] = body[ibody].density *
        (((itensor[ibody][1] + itensor[ibody][2]) * inverse_60) -
         body[ibody].volume *
             (body[ibody].xcm[1] * body[ibody].xcm[1] + body[ibody].xcm[2] * body[ibody].xcm[2]));
    tensor[1][1] = body[ibody].density *
        (((itensor[ibody][0] + itensor[ibody][2]) * inverse_60) -
         body[ibody].volume *
             (body[ibody].xcm[2] * body[ibody].xcm[2] + body[ibody].xcm[0] * body[ibody].xcm[0]));
    tensor[2][2] = body[ibody].density *
        (((itensor[ibody][0] + itensor[ibody][1]) * inverse_60) -
         body[ibody].volume *
             (body[ibody].xcm[0] * body[ibody].xcm[0] + body[ibody].xcm[1] * body[ibody].xcm[1]));

    tensor[0][1] = tensor[1][0] = -body[ibody].density *
        ((itensor[ibody][3] * inverse_120 -
          body[ibody].volume * body[ibody].xcm[0] * body[ibody].xcm[1]));
    tensor[1][2] = tensor[2][1] = -body[ibody].density *
        ((itensor[ibody][4] * inverse_120 -
          body[ibody].volume * body[ibody].xcm[1] * body[ibody].xcm[2]));
    tensor[0][2] = tensor[2][0] = -body[ibody].density *
        ((itensor[ibody][5] * inverse_120 -
          body[ibody].volume * body[ibody].xcm[2] * body[ibody].xcm[0]));

    inertia = body[ibody].inertia;
    ierror = MathEigen::jacobi3(tensor, inertia, evectors);
    if (ierror) error->all(FLERR, "Insufficient Jacobi rotations for rigid body");

    ex = body[ibody].ex_space;
    ex[0] = evectors[0][0];
    ex[1] = evectors[1][0];
    ex[2] = evectors[2][0];
    ey = body[ibody].ey_space;
    ey[0] = evectors[0][1];
    ey[1] = evectors[1][1];
    ey[2] = evectors[2][1];
    ez = body[ibody].ez_space;
    ez[0] = evectors[0][2];
    ez[1] = evectors[1][2];
    ez[2] = evectors[2][2];

    // if any principal moment < scaled EPSILON, set to 0.0

    double max;
    max = MAX(inertia[0], inertia[1]);
    max = MAX(max, inertia[2]);

    if (inertia[0] < EPSILON * max) inertia[0] = 0.0;
    if (inertia[1] < EPSILON * max) inertia[1] = 0.0;
    if (inertia[2] < EPSILON * max) inertia[2] = 0.0;

    // enforce 3 evectors as a right-handed coordinate system
    // flip 3rd vector if needed

    MathExtra::cross3(ex, ey, cross);
    if (MathExtra::dot3(cross, ez) < 0.0) MathExtra::negate3(ez);

    // create initial quaternion

    MathExtra::exyz_to_q(ex, ey, ez, body[ibody].quat);

    // convert geometric center position to principal axis coordinates
    // xcm is wrapped, but xgc is not initially
    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;
    double delta[3];
    MathExtra::sub3(xgc, xcm, delta);
    domain->minimum_image_big(FLERR, delta);
    MathExtra::transpose_matvec(ex, ey, ez, delta, body[ibody].xgc_body);
    MathExtra::add3(xcm, delta, xgc);
  }

  // forward communicate updated info of all bodies
  commflag = INITIAL;
  comm->forward_comm(this, 29);

  // displace = initial atom coords in basis of principal axes
  // set displace = 0.0 for atoms not in any rigid body


  // Also set the atom masses to sum to the mass of the body

  double qc[4], delta[3];
  double *quatatom;
  double theta_body;

  commflag = MASS_NATOMS;
  comm->forward_comm(this, 2);

  for (i = 0; i < nlocal; i++) {
    if (atom2body[i] < 0) {
      displace[i][0] = displace[i][1] = displace[i][2] = 0.0;
      continue;
    }

    Body *b = &body[atom2body[i]];

    if (rmass) {
      rmass[i] = b->mass/static_cast<double>(b->natoms);
    }
    else {
      mass[i] = b->mass/static_cast<double>(b->natoms);
    }
    
  
    xcm = b->xcm;
    delta[0] = unwrap[i][0] - xcm[0];
    delta[1] = unwrap[i][1] - xcm[1];
    delta[2] = unwrap[i][2] - xcm[2];

    MathExtra::transpose_matvec(b->ex_space, b->ey_space, b->ez_space, delta, displace[i]);

  }
  
  // also set the mass for the ghost atoms since the information should be available (this prevents excess communication)
  for (int i = nlocal; i < (nlocal + atom->nghost); i++){
    if (atom2body[i] < 0) 
      continue;
    
    Body *b = &body[atom2body[i]];
    
    if (rmass) {
      rmass[i] = b->mass/static_cast<double>(b->natoms);
    }
    else {
      mass[i] = b->mass/static_cast<double>(b->natoms);
    }
    
  }

  // forward communicate displace[i] to ghost atoms to test for valid principal moments & axes
  commflag = DISPLACE;
  comm->forward_comm(this, 3);

  // test for valid principal moments & axes
  // recompute moments of inertia around new axes
  // 3 diagonal moments should equal principal moments
  // 3 off-diagonal moments should be 0.0


  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++)
    for (i = 0; i < 6; i++) itensor[ibody][i] = 0.0;

  double tetra_volume;

  for (int n = 0; n < nanglelist; n++) {
    if (atom2body[anglelist[n][0]] < 0) continue;

    i1 = anglelist[n][0];
    i2 = anglelist[n][1];
    i3 = anglelist[n][2];

    inertia = itensor[atom2body[anglelist[n][0]]];

    inertia[0] += body[atom2body[anglelist[n][0]]].density *
        ((((((displace[i3][0] - displace[i1][0]) * (displace[i2][2] - displace[i1][2])) -
            ((displace[i2][0] - displace[i1][0]) * (displace[i3][2] - displace[i1][2]))) *
               (displace[i1][1] * (displace[i1][1] * displace[i1][1]) +
                displace[i2][1] *
                    ((displace[i1][1] * displace[i1][1]) +
                     displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                displace[i3][1] *
                    (((displace[i1][1] * displace[i1][1]) +
                      displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                     displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1]))) +
           (((displace[i2][0] - displace[i1][0]) * (displace[i3][1] - displace[i1][1])) -
            ((displace[i3][0] - displace[i1][0]) * (displace[i2][1] - displace[i1][1]))) *
               (displace[i1][2] * (displace[i1][2] * displace[i1][2]) +
                displace[i2][2] *
                    ((displace[i1][2] * displace[i1][2]) +
                     displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                displace[i3][2] *
                    (((displace[i1][2] * displace[i1][2]) +
                      displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                     displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])))) * inverse_60));
    inertia[1] += body[atom2body[anglelist[n][0]]].density *
        ((((((displace[i2][1] - displace[i1][1]) * (displace[i3][2] - displace[i1][2])) -
            ((displace[i3][1] - displace[i1][1]) * (displace[i2][2] - displace[i1][2]))) *
               (displace[i1][0] * (displace[i1][0] * displace[i1][0]) +
                displace[i2][0] *
                    ((displace[i1][0] * displace[i1][0]) +
                     displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                displace[i3][0] *
                    (((displace[i1][0] * displace[i1][0]) +
                      displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                     displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0]))) +
           (((displace[i2][0] - displace[i1][0]) * (displace[i3][1] - displace[i1][1])) -
            ((displace[i3][0] - displace[i1][0]) * (displace[i2][1] - displace[i1][1]))) *
               (displace[i1][2] * (displace[i1][2] * displace[i1][2]) +
                displace[i2][2] *
                    ((displace[i1][2] * displace[i1][2]) +
                     displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                displace[i3][2] *
                    (((displace[i1][2] * displace[i1][2]) +
                      displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                     displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])))) * inverse_60));
    inertia[2] += body[atom2body[anglelist[n][0]]].density *
        ((((((displace[i2][1] - displace[i1][1]) * (displace[i3][2] - displace[i1][2])) -
            ((displace[i3][1] - displace[i1][1]) * (displace[i2][2] - displace[i1][2]))) *
               (displace[i1][0] * (displace[i1][0] * displace[i1][0]) +
                displace[i2][0] *
                    ((displace[i1][0] * displace[i1][0]) +
                     displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                displace[i3][0] *
                    (((displace[i1][0] * displace[i1][0]) +
                      displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                     displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0]))) +
           (((displace[i3][0] - displace[i1][0]) * (displace[i2][2] - displace[i1][2])) -
            ((displace[i2][0] - displace[i1][0]) * (displace[i3][2] - displace[i1][2]))) *
               (displace[i1][1] * (displace[i1][1] * displace[i1][1]) +
                displace[i2][1] *
                    ((displace[i1][1] * displace[i1][1]) +
                     displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                displace[i3][1] *
                    (((displace[i1][1] * displace[i1][1]) +
                      displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                     displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1])))) * inverse_60));
    inertia[3] -= body[atom2body[anglelist[n][0]]].density *
        ((((((displace[i2][1] - displace[i1][1]) * (displace[i3][2] - displace[i1][2])) -
            ((displace[i3][1] - displace[i1][1]) * (displace[i2][2] - displace[i1][2]))) *
           (displace[i1][1] *
                ((((displace[i1][0] * displace[i1][0]) +
                   displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                  displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0])) +
                 displace[i1][0] *
                     (((displace[i1][0] + displace[i2][0]) + displace[i3][0]) + displace[i1][0])) +
            displace[i2][1] *
                ((((displace[i1][0] * displace[i1][0]) +
                   displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                  displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0])) +
                 displace[i2][0] *
                     (((displace[i1][0] + displace[i2][0]) + displace[i3][0]) + displace[i2][0])) +
            displace[i3][1] *
                ((((displace[i1][0] * displace[i1][0]) +
                   displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                  displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0])) +
                 displace[i3][0] *
                     (((displace[i1][0] + displace[i2][0]) + displace[i3][0]) +
                      displace[i3][0])))) * inverse_120));
    inertia[4] -= body[atom2body[anglelist[n][0]]].density *
        ((((((displace[i3][0] - displace[i1][0]) * (displace[i2][2] - displace[i1][2])) -
            ((displace[i2][0] - displace[i1][0]) * (displace[i3][2] - displace[i1][2]))) *
           (displace[i1][2] *
                ((((displace[i1][1] * displace[i1][1]) +
                   displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                  displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1])) +
                 displace[i1][1] *
                     (((displace[i1][1] + displace[i2][1]) + displace[i3][1]) + displace[i1][1])) +
            displace[i2][2] *
                ((((displace[i1][1] * displace[i1][1]) +
                   displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                  displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1])) +
                 displace[i2][1] *
                     (((displace[i1][1] + displace[i2][1]) + displace[i3][1]) + displace[i2][1])) +
            displace[i3][2] *
                ((((displace[i1][1] * displace[i1][1]) +
                   displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                  displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1])) +
                 displace[i3][1] *
                     (((displace[i1][1] + displace[i2][1]) + displace[i3][1]) +
                      displace[i3][1])))) * inverse_120));
    inertia[5] -= body[atom2body[anglelist[n][0]]].density *
        ((((((displace[i2][0] - displace[i1][0]) * (displace[i3][1] - displace[i1][1])) -
            ((displace[i3][0] - displace[i1][0]) * (displace[i2][1] - displace[i1][1]))) *
           (displace[i1][0] *
                ((((displace[i1][2] * displace[i1][2]) +
                   displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                  displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])) +
                 displace[i1][2] *
                     (((displace[i1][2] + displace[i2][2]) + displace[i3][2]) + displace[i1][2])) +
            displace[i2][0] *
                ((((displace[i1][2] * displace[i1][2]) +
                   displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                  displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])) +
                 displace[i2][2] *
                     (((displace[i1][2] + displace[i2][2]) + displace[i3][2]) + displace[i2][2])) +
            displace[i3][0] *
                ((((displace[i1][2] * displace[i1][2]) +
                   displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                  displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])) +
                 displace[i3][2] *
                     (((displace[i1][2] + displace[i2][2]) + displace[i3][2]) +
                      displace[i3][2])))) * inverse_120));
  }

  // reverse communicate inertia tensor of all bodies

  commflag = ITENSOR;
  comm->reverse_comm(this, 6);

  double inverse_norm;
  for (ibody = 0; ibody < nlocal_body; ibody++) {
    inertia = body[ibody].inertia;

    if (inertia[0] == 0.0) {
      if (fabs(itensor[ibody][0]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][0] - inertia[0]) / inertia[0]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    }
    if (inertia[1] == 0.0) {
      if (fabs(itensor[ibody][1]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][1] - inertia[1]) / inertia[1]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    }
    if (inertia[2] == 0.0) {
      if (fabs(itensor[ibody][2]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][2] - inertia[2]) / inertia[2]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    }
    inverse_norm = 1.0 / ((inertia[0] + inertia[1] + inertia[2]) * (THIRD));
    if (fabs(itensor[ibody][3] * inverse_norm) > TOLERANCE ||
        fabs(itensor[ibody][4] * inverse_norm) > TOLERANCE || fabs(itensor[ibody][5] * inverse_norm) > TOLERANCE)
      error->all(FLERR, "Fix rigid: Bad principal moments");
  }

  // clean up
  memory->destroy(itensor);
}

/* ----------------------------------------------------------------------
   Recalculation of Abraded Bodies' COM, Volume, and Inertia
------------------------------------------------------------------------- */

void FixRigidAbrade::resetup_bodies_static()
{

  int i, ibody;
  int nlocal = atom->nlocal;
  double *rmass = atom->rmass;
  double *mass = atom->mass;


  // acquire ghost bodies via forward comm
  // set atom2body for ghost atoms via forward comm
  // set atom2body for other owned atoms via reset_atom2body()

  nghost_body = 0;
  commflag = FULL_BODY;
  comm->forward_comm(this);

  // NOTE:: reset_atom2body() isn't called here which provides a slight performance gain without changing the the simulated output for tested cases.
  // This may need more extensive testing incase there is a situation where the atom2body can change for ghost atoms since last setup the body.

  // reset_atom2body();

  // compute mass & center-of-mass of each rigid body
  double **x = atom->x;
  double *xcm;
  double *xgc;

  if (proc_abraded_flag)
  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[ibody].abraded_flag) continue;

    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;
    xcm[0] = xcm[1] = xcm[2] = 0.0;
    xgc[0] = xgc[1] = xgc[2] = 0.0;
    body[ibody].mass = 0.0;
    body[ibody].volume = 0.0;
    body[ibody].density = density;
    body[ibody].natoms = 0;
  }

  double massone;

  // Cycling through the local atoms and storing their unwrapped coordinates.
  // Additionally, increment the number of atoms in their respective body
  if (proc_abraded_flag)
  for (i = 0; i < nlocal; i++) {

    if (atom2body[i] < 0) continue;

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[atom2body[i]].abraded_flag) continue;

    // Calculated unwrapped coords of all local atoms in bodies
    domain->unmap(x[i], xcmimage[i], unwrap[i]);

    Body *b = &body[atom2body[i]];
    b->natoms++;
  }

  int nanglelist = neighbor->nanglelist;
  int **anglelist = neighbor->anglelist;
  int i1, i2, i3;

  // communicate unwrapped position of owned atoms to ghost atoms
  commflag = UNWRAP;
  comm->forward_comm(this, 3);

  // Calculating body volume, mass and COM from constituent tetrahedra

  double inverse_24 = 1.0/24.0;
  double inverse_6 = 1.0/6.0;

  if (proc_abraded_flag)
  for (int n = 0; n < (neighbor->nanglelist + remesh_overflow_threshold + overflow_anglelist.size()); n++) {

    // Storing each atom in the angle
    if (n + 1 <= (nanglelist + remesh_overflow_threshold)) {
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];
    } else {
      // reading angles from the overflow list populated during remeshing. This will be cleared and angles properly added to neighbor->anglelist during end_of_step()
      i1 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][0];
      i2 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][1];
      i3 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][2];
    }

    // Only process the angle belongs to a body
    if (atom2body[i1] < 0) continue;

    // Only process if the owning body has changed shape
    if (!body[atom2body[i1]].abraded_flag) continue;

    Body *b = &body[atom2body[i1]];

    xcm = b->xcm;
    xgc = b->xgc;

    b->volume += ((((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
                   ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
                  ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])) * inverse_6;

    xcm[0] +=
        ((((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
          ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
         (((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
          unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]))) * inverse_24;
    xcm[1] +=
        ((((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][2] - unwrap[i1][2])) -
          ((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][2] - unwrap[i1][2]))) *
         (((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
          unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]))) * inverse_24;
    xcm[2] +=
        ((((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][1] - unwrap[i1][1])) -
          ((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][1] - unwrap[i1][1]))) *
         (((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
          unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]))) * inverse_24;
    xgc[0] +=
        ((((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
          ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
         (((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
          unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]))) * inverse_24;
    xgc[1] +=
        ((((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][2] - unwrap[i1][2])) -
          ((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][2] - unwrap[i1][2]))) *
         (((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
          unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]))) * inverse_24;
    xgc[2] +=
        ((((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][1] - unwrap[i1][1])) -
          ((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][1] - unwrap[i1][1]))) *
         (((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
          unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]))) * inverse_24;
  }

  // reverse communicate xcm, mass of all bodies
  commflag = XCM_MASS;
  comm->reverse_comm(this, 9);

  double inverse_volume;

  if (proc_abraded_flag)
  for (ibody = 0; ibody < nlocal_body; ibody++) {

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[ibody].abraded_flag) continue;

    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;

    // Setting each bodies' COM
    inverse_volume = 1.0 /  body[ibody].volume;
    xcm[0] *= inverse_volume;
    xcm[1] *= inverse_volume;
    xcm[2] *= inverse_volume;
    xgc[0] *= inverse_volume;
    xgc[1] *= inverse_volume;
    xgc[2] *= inverse_volume;

    // Setting the mass of each body
    body[ibody].mass = body[ibody].volume * density;
    
    if (equalise_surface_flag) {
      body[ibody].initial_volume = body[ibody].volume;
    }
    
    body[ibody].abraded_volume = body[ibody].initial_volume - body[ibody].volume;

    // Positioning the owning atom at the COM
    x[body[ibody].ilocal][0] = body[ibody].xcm[0];
    x[body[ibody].ilocal][1] = body[ibody].xcm[1];
    x[body[ibody].ilocal][2] = body[ibody].xcm[2];
    unwrap[body[ibody].ilocal][0] = body[ibody].xcm[0];
    unwrap[body[ibody].ilocal][1] = body[ibody].xcm[1];
    unwrap[body[ibody].ilocal][2] = body[ibody].xcm[2];

  }

  // compute 6 moments of inertia of each body in Cartesian reference frame
  // dx,dy,dz = coords relative to center-of-mass
  // symmetric 3x3 inertia tensor stored in Voigt notation as 6-vector

  // also place owning atom at COM following the mapping of XCM in pre_neighbor

  memory->create(itensor, nlocal_body + nghost_body, 6, "rigid/abrade:itensor");
  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    for (i = 0; i < 6; i++) itensor[ibody][i] = 0.0;
  }

  double dx, dy, dz;
  double *inertia;
  
  if (proc_abraded_flag)
  for (int n = 0; n < (neighbor->nanglelist + remesh_overflow_threshold + overflow_anglelist.size()); n++) {

    // Storing each atom in the angle
    if (n + 1 <= (nanglelist + remesh_overflow_threshold)) {
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];
    } else {
      // reading angles from the overflow list populated during remeshing. This will be cleared and angles properly added to neighbor->anglelist during end_of_step()
      i1 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][0];
      i2 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][1];
      i3 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][2];
    }

    if (atom2body[i1] < 0) continue;

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[atom2body[i1]].abraded_flag) continue;

    Body *b = &body[atom2body[i1]];

    inertia = itensor[atom2body[i1]];
    inertia[0] += (((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
                   ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
        (unwrap[i1][0] * (unwrap[i1][0] * unwrap[i1][0]) +
         unwrap[i2][0] *
             ((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
         unwrap[i3][0] *
             (((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
              unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])));
    inertia[1] += (((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][2] - unwrap[i1][2])) -
                   ((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][2] - unwrap[i1][2]))) *
        (unwrap[i1][1] * (unwrap[i1][1] * unwrap[i1][1]) +
         unwrap[i2][1] *
             ((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
         unwrap[i3][1] *
             (((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
              unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1])));
    inertia[2] += (((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][1] - unwrap[i1][1])) -
                   ((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][1] - unwrap[i1][1]))) *
        (unwrap[i1][2] * (unwrap[i1][2] * unwrap[i1][2]) +
         unwrap[i2][2] *
             ((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
         unwrap[i3][2] *
             (((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
              unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2])));
    inertia[3] += (((unwrap[i2][1] - unwrap[i1][1]) * (unwrap[i3][2] - unwrap[i1][2])) -
                   ((unwrap[i3][1] - unwrap[i1][1]) * (unwrap[i2][2] - unwrap[i1][2]))) *
        (unwrap[i1][1] *
             ((((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
               unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])) +
              unwrap[i1][0] * (((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]) + unwrap[i1][0])) +
         unwrap[i2][1] *
             ((((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
               unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])) +
              unwrap[i2][0] * (((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]) + unwrap[i2][0])) +
         unwrap[i3][1] *
             ((((unwrap[i1][0] * unwrap[i1][0]) + unwrap[i2][0] * (unwrap[i1][0] + unwrap[i2][0])) +
               unwrap[i3][0] * ((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0])) +
              unwrap[i3][0] * (((unwrap[i1][0] + unwrap[i2][0]) + unwrap[i3][0]) + unwrap[i3][0])));
    inertia[4] += (((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][2] - unwrap[i1][2])) -
                   ((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][2] - unwrap[i1][2]))) *
        (unwrap[i1][2] *
             ((((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
               unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1])) +
              unwrap[i1][1] * (((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]) + unwrap[i1][1])) +
         unwrap[i2][2] *
             ((((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
               unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1])) +
              unwrap[i2][1] * (((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]) + unwrap[i2][1])) +
         unwrap[i3][2] *
             ((((unwrap[i1][1] * unwrap[i1][1]) + unwrap[i2][1] * (unwrap[i1][1] + unwrap[i2][1])) +
               unwrap[i3][1] * ((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1])) +
              unwrap[i3][1] * (((unwrap[i1][1] + unwrap[i2][1]) + unwrap[i3][1]) + unwrap[i3][1])));
    inertia[5] += (((unwrap[i2][0] - unwrap[i1][0]) * (unwrap[i3][1] - unwrap[i1][1])) -
                   ((unwrap[i3][0] - unwrap[i1][0]) * (unwrap[i2][1] - unwrap[i1][1]))) *
        (unwrap[i1][0] *
             ((((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
               unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2])) +
              unwrap[i1][2] * (((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]) + unwrap[i1][2])) +
         unwrap[i2][0] *
             ((((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
               unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2])) +
              unwrap[i2][2] * (((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]) + unwrap[i2][2])) +
         unwrap[i3][0] *
             ((((unwrap[i1][2] * unwrap[i1][2]) + unwrap[i2][2] * (unwrap[i1][2] + unwrap[i2][2])) +
               unwrap[i3][2] * ((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2])) +
              unwrap[i3][2] * (((unwrap[i1][2] + unwrap[i2][2]) + unwrap[i3][2]) + unwrap[i3][2])));
  }

  // reverse communicate inertia tensor of all bodies

  commflag = ITENSOR;
  comm->reverse_comm(this, 6);

  // diagonalize inertia tensor for each body via Jacobi rotations
  // inertia = 3 eigenvalues = principal moments of inertia
  // evectors and exzy_space = 3 evectors = principal axes of rigid body

  int ierror;
  double cross[3];
  double tensor[3][3], evectors[3][3];
  double *ex, *ey, *ez;

  double inverse_60 = 1.0 / 60.0;
  double inverse_120 = 1.0 / 120.0;
  if (proc_abraded_flag)
  for (ibody = 0; ibody < nlocal_body; ibody++) {

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[ibody].abraded_flag) continue;

    tensor[0][0] = body[ibody].density *
        (((itensor[ibody][1] + itensor[ibody][2]) * inverse_60) -
         body[ibody].volume *
             (body[ibody].xcm[1] * body[ibody].xcm[1] + body[ibody].xcm[2] * body[ibody].xcm[2]));
    tensor[1][1] = body[ibody].density *
        (((itensor[ibody][0] + itensor[ibody][2]) * inverse_60) -
         body[ibody].volume *
             (body[ibody].xcm[2] * body[ibody].xcm[2] + body[ibody].xcm[0] * body[ibody].xcm[0]));
    tensor[2][2] = body[ibody].density *
        (((itensor[ibody][0] + itensor[ibody][1]) * inverse_60) -
         body[ibody].volume *
             (body[ibody].xcm[0] * body[ibody].xcm[0] + body[ibody].xcm[1] * body[ibody].xcm[1]));

    tensor[0][1] = tensor[1][0] = -body[ibody].density *
        ((itensor[ibody][3] * inverse_120 -
          body[ibody].volume * body[ibody].xcm[0] * body[ibody].xcm[1]));
    tensor[1][2] = tensor[2][1] = -body[ibody].density *
        ((itensor[ibody][4] * inverse_120 -
          body[ibody].volume * body[ibody].xcm[1] * body[ibody].xcm[2]));
    tensor[0][2] = tensor[2][0] = -body[ibody].density *
        ((itensor[ibody][5] * inverse_120 -
          body[ibody].volume * body[ibody].xcm[2] * body[ibody].xcm[0]));

    inertia = body[ibody].inertia;
    ierror = MathEigen::jacobi3(tensor, inertia, evectors);
    if (ierror) error->all(FLERR, "Insufficient Jacobi rotations for rigid body");

    ex = body[ibody].ex_space;
    ex[0] = evectors[0][0];
    ex[1] = evectors[1][0];
    ex[2] = evectors[2][0];
    ey = body[ibody].ey_space;
    ey[0] = evectors[0][1];
    ey[1] = evectors[1][1];
    ey[2] = evectors[2][1];
    ez = body[ibody].ez_space;
    ez[0] = evectors[0][2];
    ez[1] = evectors[1][2];
    ez[2] = evectors[2][2];

    // if any principal moment < scaled EPSILON, set to 0.0

    double max;
    max = MAX(inertia[0], inertia[1]);
    max = MAX(max, inertia[2]);

    if (inertia[0] < EPSILON * max) inertia[0] = 0.0;
    if (inertia[1] < EPSILON * max) inertia[1] = 0.0;
    if (inertia[2] < EPSILON * max) inertia[2] = 0.0;

    // enforce 3 evectors as a right-handed coordinate system
    // flip 3rd vector if needed

    MathExtra::cross3(ex, ey, cross);
    if (MathExtra::dot3(cross, ez) < 0.0) MathExtra::negate3(ez);

    // create initial quaternion

    MathExtra::exyz_to_q(ex, ey, ez, body[ibody].quat);

    // convert geometric center position to principal axis coordinates
    // xcm is wrapped, but xgc is not initially
    xcm = body[ibody].xcm;
    xgc = body[ibody].xgc;
    double delta[3];
    MathExtra::sub3(xgc, xcm, delta);
    domain->minimum_image_big(FLERR, delta);
    MathExtra::transpose_matvec(ex, ey, ez, delta, body[ibody].xgc_body);
    MathExtra::add3(xcm, delta, xgc);
  }

  // forward communicate updated info of all bodies

  commflag = INITIAL;
  comm->forward_comm(this, 29);

  // displace = initial atom coords in basis of principal axes
  // set displace = 0.0 for atoms not in any rigid body

  // displace[i] must be recalculated since the bodies' coordinate system may have changed with the updating of COM and the prinicpal axes
  // also set the update the mass of atoms

  double qc[4], delta[3];
  double *quatatom;
  double theta_body;

  commflag = MASS_NATOMS;
  comm->forward_comm(this, 2);
  
  if (proc_abraded_flag)
  for (i = 0; i < nlocal; i++) {

    if (atom2body[i] < 0) {
      displace[i][0] = displace[i][1] = displace[i][2] = 0.0;
      continue;
    }

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[atom2body[i]].abraded_flag) continue;

    Body *b = &body[atom2body[i]];

    xcm = b->xcm;
    delta[0] = unwrap[i][0] - xcm[0];
    delta[1] = unwrap[i][1] - xcm[1];
    delta[2] = unwrap[i][2] - xcm[2];
    
    if (rmass) {
      rmass[i] = b->mass/static_cast<double>(b->natoms);
    }
    else {
      mass[i] = b->mass/static_cast<double>(b->natoms);
    }

    MathExtra::transpose_matvec(b->ex_space, b->ey_space, b->ez_space, delta, displace[i]);

  }

  // also set the mass for the ghost atoms since the information should be available (this prevents excess communication)

  for (int i = nlocal; i < (nlocal + atom->nghost); i++){
    if (atom2body[i] < 0) continue;
    
    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[ibody].abraded_flag) continue;

    Body *b = &body[atom2body[i]];
    
    if (rmass) {
      rmass[i] = b->mass / static_cast<double>(b->natoms);
    }
    else { 
      mass[i] = b->mass / static_cast<double>(b->natoms);
    }
    
  }

  // forward communicate displace[i] to ghost atoms to test for valid principal moments & axes
  commflag = DISPLACE;
  comm->forward_comm(this, 3);

  // test for valid principal moments & axes
  // recompute moments of inertia around new axes
  // 3 diagonal moments should equal principal moments
  // 3 off-diagonal moments should be 0.0

  
  if (proc_abraded_flag)
  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[ibody].abraded_flag) continue;
    for (i = 0; i < 6; i++) itensor[ibody][i] = 0.0;
  }

  if (proc_abraded_flag)
  for (int n = 0; n < (neighbor->nanglelist + remesh_overflow_threshold + overflow_anglelist.size()); n++) {

    // Storing each atom in the angle
    if (n + 1 <= (nanglelist + remesh_overflow_threshold)) {
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];
    } else {
      // reading angles from the overflow list populated during remeshing. This will be cleared and angles properly added to neighbor->anglelist during end_of_step()
      i1 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][0];
      i2 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][1];
      i3 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][2];
    }

    if (atom2body[i1] < 0) continue;

    // Only processing properties relevant to bodies which have abraded and changed shape
    if (!body[atom2body[i1]].abraded_flag) continue;

    inertia = itensor[atom2body[i1]];

    inertia[0] += body[atom2body[i1]].density *
        ((((((displace[i3][0] - displace[i1][0]) * (displace[i2][2] - displace[i1][2])) -
            ((displace[i2][0] - displace[i1][0]) * (displace[i3][2] - displace[i1][2]))) *
               (displace[i1][1] * (displace[i1][1] * displace[i1][1]) +
                displace[i2][1] *
                    ((displace[i1][1] * displace[i1][1]) +
                     displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                displace[i3][1] *
                    (((displace[i1][1] * displace[i1][1]) +
                      displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                     displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1]))) +
           (((displace[i2][0] - displace[i1][0]) * (displace[i3][1] - displace[i1][1])) -
            ((displace[i3][0] - displace[i1][0]) * (displace[i2][1] - displace[i1][1]))) *
               (displace[i1][2] * (displace[i1][2] * displace[i1][2]) +
                displace[i2][2] *
                    ((displace[i1][2] * displace[i1][2]) +
                     displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                displace[i3][2] *
                    (((displace[i1][2] * displace[i1][2]) +
                      displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                     displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])))) * inverse_60));
    inertia[1] += body[atom2body[i1]].density *
        ((((((displace[i2][1] - displace[i1][1]) * (displace[i3][2] - displace[i1][2])) -
            ((displace[i3][1] - displace[i1][1]) * (displace[i2][2] - displace[i1][2]))) *
               (displace[i1][0] * (displace[i1][0] * displace[i1][0]) +
                displace[i2][0] *
                    ((displace[i1][0] * displace[i1][0]) +
                     displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                displace[i3][0] *
                    (((displace[i1][0] * displace[i1][0]) +
                      displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                     displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0]))) +
           (((displace[i2][0] - displace[i1][0]) * (displace[i3][1] - displace[i1][1])) -
            ((displace[i3][0] - displace[i1][0]) * (displace[i2][1] - displace[i1][1]))) *
               (displace[i1][2] * (displace[i1][2] * displace[i1][2]) +
                displace[i2][2] *
                    ((displace[i1][2] * displace[i1][2]) +
                     displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                displace[i3][2] *
                    (((displace[i1][2] * displace[i1][2]) +
                      displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                     displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])))) * inverse_60));
    inertia[2] += body[atom2body[i1]].density *
        ((((((displace[i2][1] - displace[i1][1]) * (displace[i3][2] - displace[i1][2])) -
            ((displace[i3][1] - displace[i1][1]) * (displace[i2][2] - displace[i1][2]))) *
               (displace[i1][0] * (displace[i1][0] * displace[i1][0]) +
                displace[i2][0] *
                    ((displace[i1][0] * displace[i1][0]) +
                     displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                displace[i3][0] *
                    (((displace[i1][0] * displace[i1][0]) +
                      displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                     displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0]))) +
           (((displace[i3][0] - displace[i1][0]) * (displace[i2][2] - displace[i1][2])) -
            ((displace[i2][0] - displace[i1][0]) * (displace[i3][2] - displace[i1][2]))) *
               (displace[i1][1] * (displace[i1][1] * displace[i1][1]) +
                displace[i2][1] *
                    ((displace[i1][1] * displace[i1][1]) +
                     displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                displace[i3][1] *
                    (((displace[i1][1] * displace[i1][1]) +
                      displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                     displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1])))) * inverse_60));
    inertia[3] -= body[atom2body[i1]].density *
        ((((((displace[i2][1] - displace[i1][1]) * (displace[i3][2] - displace[i1][2])) -
            ((displace[i3][1] - displace[i1][1]) * (displace[i2][2] - displace[i1][2]))) *
           (displace[i1][1] *
                ((((displace[i1][0] * displace[i1][0]) +
                   displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                  displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0])) +
                 displace[i1][0] *
                     (((displace[i1][0] + displace[i2][0]) + displace[i3][0]) + displace[i1][0])) +
            displace[i2][1] *
                ((((displace[i1][0] * displace[i1][0]) +
                   displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                  displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0])) +
                 displace[i2][0] *
                     (((displace[i1][0] + displace[i2][0]) + displace[i3][0]) + displace[i2][0])) +
            displace[i3][1] *
                ((((displace[i1][0] * displace[i1][0]) +
                   displace[i2][0] * (displace[i1][0] + displace[i2][0])) +
                  displace[i3][0] * ((displace[i1][0] + displace[i2][0]) + displace[i3][0])) +
                 displace[i3][0] *
                     (((displace[i1][0] + displace[i2][0]) + displace[i3][0]) +
                      displace[i3][0])))) * inverse_120));
    inertia[4] -= body[atom2body[i1]].density *
        ((((((displace[i3][0] - displace[i1][0]) * (displace[i2][2] - displace[i1][2])) -
            ((displace[i2][0] - displace[i1][0]) * (displace[i3][2] - displace[i1][2]))) *
           (displace[i1][2] *
                ((((displace[i1][1] * displace[i1][1]) +
                   displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                  displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1])) +
                 displace[i1][1] *
                     (((displace[i1][1] + displace[i2][1]) + displace[i3][1]) + displace[i1][1])) +
            displace[i2][2] *
                ((((displace[i1][1] * displace[i1][1]) +
                   displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                  displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1])) +
                 displace[i2][1] *
                     (((displace[i1][1] + displace[i2][1]) + displace[i3][1]) + displace[i2][1])) +
            displace[i3][2] *
                ((((displace[i1][1] * displace[i1][1]) +
                   displace[i2][1] * (displace[i1][1] + displace[i2][1])) +
                  displace[i3][1] * ((displace[i1][1] + displace[i2][1]) + displace[i3][1])) +
                 displace[i3][1] *
                     (((displace[i1][1] + displace[i2][1]) + displace[i3][1]) +
                      displace[i3][1])))) * inverse_120));
    inertia[5] -= body[atom2body[i1]].density *
        ((((((displace[i2][0] - displace[i1][0]) * (displace[i3][1] - displace[i1][1])) -
            ((displace[i3][0] - displace[i1][0]) * (displace[i2][1] - displace[i1][1]))) *
           (displace[i1][0] *
                ((((displace[i1][2] * displace[i1][2]) +
                   displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                  displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])) +
                 displace[i1][2] *
                     (((displace[i1][2] + displace[i2][2]) + displace[i3][2]) + displace[i1][2])) +
            displace[i2][0] *
                ((((displace[i1][2] * displace[i1][2]) +
                   displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                  displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])) +
                 displace[i2][2] *
                     (((displace[i1][2] + displace[i2][2]) + displace[i3][2]) + displace[i2][2])) +
            displace[i3][0] *
                ((((displace[i1][2] * displace[i1][2]) +
                   displace[i2][2] * (displace[i1][2] + displace[i2][2])) +
                  displace[i3][2] * ((displace[i1][2] + displace[i2][2]) + displace[i3][2])) +
                 displace[i3][2] *
                     (((displace[i1][2] + displace[i2][2]) + displace[i3][2]) +
                      displace[i3][2])))) * inverse_120));
  }

  // reverse communicate inertia tensor of all bodies

  commflag = ITENSOR;
  comm->reverse_comm(this, 6);

  // error check that re-computed moments of inertia match diagonalized ones
  double inverse_norm;
  
  if (proc_abraded_flag)
  for (ibody = 0; ibody < nlocal_body; ibody++) {

    // Only performing error checks on bodies that have been updated
    if (!body[ibody].abraded_flag) continue;

    inertia = body[ibody].inertia;

    if (inertia[0] == 0.0) {
      if (fabs(itensor[ibody][0]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][0] - inertia[0]) / inertia[0]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    }
    if (inertia[1] == 0.0) {
      if (fabs(itensor[ibody][1]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][1] - inertia[1]) / inertia[1]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    }
    if (inertia[2] == 0.0) {
      if (fabs(itensor[ibody][2]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    } else {
      if (fabs((itensor[ibody][2] - inertia[2]) / inertia[2]) > TOLERANCE)
        error->all(FLERR, "Fix rigid: Bad principal moments");
    }
    inverse_norm = 1.0 / ((inertia[0] + inertia[1] + inertia[2]) * (THIRD));
    if (fabs(itensor[ibody][3] * inverse_norm) > TOLERANCE || fabs(itensor[ibody][4] * inverse_norm) > TOLERANCE ||
        fabs(itensor[ibody][5] * inverse_norm) > TOLERANCE)
      error->all(FLERR, "Fix rigid: Bad principal moments");
  }

  // clean up
  memory->destroy(itensor);
}

/* ----------------------------------------------------------------------
   one-time initialization of dynamic rigid body attributes
   vcm and angmom, computed explicitly from constituent particles
   not done if body properties read from file, e.g. for overlapping particles
------------------------------------------------------------------------- */

void FixRigidAbrade::setup_bodies_dynamic()
{
  int i, ibody;
  double massone, radone;

  // sum vcm, angmom across all rigid bodies
  // vcm = velocity of COM
  // angmom = angular momentum around COM

  double **v = atom->v;
  double *xcm, *vcm, *acm;

  // Zeroing body properties
  for (ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    vcm = body[ibody].vcm;
    vcm[0] = vcm[1] = vcm[2] = 0.0;
    acm = body[ibody].angmom;
    acm[0] = acm[1] = acm[2] = 0.0;
  }

  int **anglelist = neighbor->anglelist;
  int nanglelist = neighbor->nanglelist;
  double dx1, dx2, dx3, dy1, dy2, dy3, dz1, dz2, dz3, dx4, dy4, dz4;
  double J[3][3];
  double det_J;
  double v_centroid[3];
  double omega_centroid[3];
  double tetra_volume;
  double tetra_mass;
  double Ixx, Iyy, Izz, Iyz, Ixz, Ixy;
  double tetra_tensor[3][3];
  double r_centroid[3];
  double r_len;
  int i1, i2, i3;

  // Cycling through angle list to consider the tetrahedra constructed about the COM
  for (int n = 0;
       n < (neighbor->nanglelist + remesh_overflow_threshold + overflow_anglelist.size()); n++) {

    // Storing each atom in the angle
    if (n + 1 <= (nanglelist + remesh_overflow_threshold)) {
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];
    } else {
      // reading angles from the overflow list populated during remeshing. This will be cleared and angles properly added to neighbor->anglelist during end_of_step()
      i1 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][0];
      i2 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][1];
      i3 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][2];
    }

    if (atom2body[i1] < 0) continue;
    Body *b = &body[atom2body[i1]];

    // Offsetting the atom positions about the bodies' COM
    xcm = b->xcm;
    dx1 = unwrap[i1][0] - xcm[0];
    dy1 = unwrap[i1][1] - xcm[1];
    dz1 = unwrap[i1][2] - xcm[2];    // i1 position
    dx2 = unwrap[i2][0] - xcm[0];
    dy2 = unwrap[i2][1] - xcm[1];
    dz2 = unwrap[i2][2] - xcm[2];    // i2 position
    dx3 = unwrap[i3][0] - xcm[0];
    dy3 = unwrap[i3][1] - xcm[1];
    dz3 = unwrap[i3][2] - xcm[2];    // i3 position
    dx4 = 0.0;
    dy4 = 0.0;
    dz4 = 0.0;    // COM position (0.0, 0.0, 0.0)

    // Calculating the determinat of the offset coordinates (equals 6*volume of the tetrahedra)
    J[0][0] = dx2 - dx1;
    J[0][1] = dx3 - dx1;
    J[0][2] = dx4 - dx1;
    J[1][0] = dy2 - dy1;
    J[1][1] = dy3 - dy1;
    J[1][2] = dy4 - dy1;
    J[2][0] = dz2 - dz1;
    J[2][1] = dz3 - dz1;
    J[2][2] = dz4 - dz1;
    det_J = MathExtra::det3(J);

    tetra_volume = -(det_J / 6.0);
    tetra_mass = b->density * tetra_volume;

    v_centroid[0] = ((v[i1][0] + v[i2][0] + v[i3][0]) * (THIRD));
    v_centroid[1] = ((v[i1][1] + v[i2][1] + v[i3][1]) * (THIRD));
    v_centroid[2] = ((v[i1][2] + v[i2][2] + v[i3][2]) * (THIRD));

    vcm = b->vcm;
    vcm[0] += tetra_mass * v_centroid[0] * dynamic_flag;
    vcm[1] += tetra_mass * v_centroid[1] * dynamic_flag;
    vcm[2] += tetra_mass * v_centroid[2] * dynamic_flag;
  }

  // reverse communicate vcm of all bodies

  commflag = VCM;
  comm->reverse_comm(this, 3);

  // divide velocity of COM by the mass
  for (ibody = 0; ibody < nlocal_body; ibody++) {
    vcm = body[ibody].vcm;
    vcm[0] /= body[ibody].mass;
    vcm[1] /= body[ibody].mass;
    vcm[2] /= body[ibody].mass;
  }

  // forward communicate vcm of all bodies since it is used to setup the acm of ghost angles
  commflag = VCM;
  comm->forward_comm(this, 3);

  // Setting angular momentum working wholly in the global coordinate space
  for (int n = 0;
       n < (neighbor->nanglelist + remesh_overflow_threshold + overflow_anglelist.size()); n++) {

    // Storing each atom in the angle
    if (n + 1 <= (nanglelist + remesh_overflow_threshold)) {
      i1 = anglelist[n][0];
      i2 = anglelist[n][1];
      i3 = anglelist[n][2];
    } else {
      // reading angles from the overflow list populated during remeshing. This will be cleared and angles properly added to neighbor->anglelist during end_of_step()
      i1 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][0];
      i2 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][1];
      i3 = overflow_anglelist[n - (nanglelist + remesh_overflow_threshold)][2];
    }

    if (atom2body[i1] < 0) continue;
    Body *b = &body[atom2body[i1]];

    // Offsetting the atom positions about the bodies' COM
    xcm = b->xcm;
    dx1 = unwrap[i1][0] - xcm[0];
    dy1 = unwrap[i1][1] - xcm[1];
    dz1 = unwrap[i1][2] - xcm[2];    // i1 position
    dx2 = unwrap[i2][0] - xcm[0];
    dy2 = unwrap[i2][1] - xcm[1];
    dz2 = unwrap[i2][2] - xcm[2];    // i2 position
    dx3 = unwrap[i3][0] - xcm[0];
    dy3 = unwrap[i3][1] - xcm[1];
    dz3 = unwrap[i3][2] - xcm[2];    // i3 position
    dx4 = 0.0;
    dy4 = 0.0;
    dz4 = 0.0;    // COM position (0.0, 0.0, 0.0)

    // Calculating the determinat of the offset coordinates (equals 6*volume of the tetrahedra)
    J[0][0] = dx2 - dx1;
    J[0][1] = dx3 - dx1;
    J[0][2] = dx4 - dx1;
    J[1][0] = dy2 - dy1;
    J[1][1] = dy3 - dy1;
    J[1][2] = dy4 - dy1;
    J[2][0] = dz2 - dz1;
    J[2][1] = dz3 - dz1;
    J[2][2] = dz4 - dz1;
    det_J = MathExtra::det3(J);

    Ixx = b->density * (-det_J) *
        (dy1 * dy1 + dy1 * dy2 + dy2 * dy2 + dy1 * dy3 + dy2 * dy3 + dy3 * dy3 + dz1 * dz1 +
         dz1 * dz2 + dz2 * dz2 + dz1 * dz3 + dz2 * dz3 + dz3 * dz3) /
        60.0;
    Iyy = b->density * (-det_J) *
        (dx1 * dx1 + dx1 * dx2 + dx2 * dx2 + dx1 * dx3 + dx2 * dx3 + dx3 * dx3 + dz1 * dz1 +
         dz1 * dz2 + dz2 * dz2 + dz1 * dz3 + dz2 * dz3 + dz3 * dz3) /
        60.0;
    Izz = b->density * (-det_J) *
        (dx1 * dx1 + dx1 * dx2 + dx2 * dx2 + dx1 * dx3 + dx2 * dx3 + dx3 * dx3 + dy1 * dy1 +
         dy1 * dy2 + dy2 * dy2 + dy1 * dy3 + dy2 * dy3 + dy3 * dy3) /
        60.0;

    Iyz = -b->density * (-det_J) *
        (2 * dy1 * dz1 + dy2 * dz1 + dy3 * dz1 + dy1 * dz2 + 2 * dy2 * dz2 + dy3 * dz2 + dy1 * dz3 +
         dy2 * dz3 + 2 * dy3 * dz3) /
        120.0;
    Ixz = -b->density * (-det_J) *
        (2 * dx1 * dz1 + dx2 * dz1 + dx3 * dz1 + dx1 * dz2 + 2 * dx2 * dz2 + dx3 * dz2 + dx1 * dz3 +
         dx2 * dz3 + 2 * dx3 * dz3) /
        120.0;
    Ixy = -b->density * (-det_J) *
        (2 * dx1 * dy1 + dx2 * dy1 + dx3 * dy1 + dx1 * dy2 + 2 * dx2 * dy2 + dx3 * dy2 + dx1 * dy3 +
         dx2 * dy3 + 2 * dx3 * dy3) /
        120.0;

    tetra_tensor[0][0] = Ixx;
    tetra_tensor[1][1] = Iyy;
    tetra_tensor[2][2] = Izz;
    tetra_tensor[0][1] = tetra_tensor[1][0] = Ixy;
    tetra_tensor[0][2] = tetra_tensor[2][0] = Ixz;
    tetra_tensor[1][2] = tetra_tensor[2][1] = Iyz;

    // Converting from a linear velocity to an orbital angular velocity about the COM (r x v)/ |r||r|
    vcm = b->vcm;
    v_centroid[0] = (((v[i1][0] + v[i2][0] + v[i3][0]) * (THIRD)) - vcm[0]);
    v_centroid[1] = (((v[i1][1] + v[i2][1] + v[i3][1]) * (THIRD)) - vcm[1]);
    v_centroid[2] = (((v[i1][2] + v[i2][2] + v[i3][2]) * (THIRD)) - vcm[2]);

    r_centroid[0] = (dx1 + dx2 + dx3) * (THIRD);
    r_centroid[1] = (dy1 + dy2 + dy3) * (THIRD);
    r_centroid[2] = (dz1 + dz2 + dz3) * (THIRD);
    r_len = MathExtra::len3(r_centroid);

    MathExtra::cross3(r_centroid, v_centroid, omega_centroid);
    omega_centroid[0] = omega_centroid[0] / (r_len * r_len);
    omega_centroid[1] = omega_centroid[1] / (r_len * r_len);
    omega_centroid[2] = omega_centroid[2] / (r_len * r_len);

    acm = b->angmom;
    acm[0] += ((tetra_tensor[0][0] * omega_centroid[0]) + (tetra_tensor[0][1] * omega_centroid[1]) +
               (tetra_tensor[0][2] * omega_centroid[2])) *
        dynamic_flag;
    acm[1] += ((tetra_tensor[1][0] * omega_centroid[0]) + (tetra_tensor[1][1] * omega_centroid[1]) +
               (tetra_tensor[1][2] * omega_centroid[2])) *
        dynamic_flag;
    acm[2] += ((tetra_tensor[2][0] * omega_centroid[0]) + (tetra_tensor[2][1] * omega_centroid[1]) +
               (tetra_tensor[2][2] * omega_centroid[2])) *
        dynamic_flag;
  }

  // reverse communicate angmom of all bodies
  commflag = ANGMOM;
  comm->reverse_comm(this, 3);

  // setting values for ghost bodies to 0.
  for (int ibody = nlocal_body; ibody < (nlocal_body + nghost_body); ibody++) {
    body[ibody].vcm[0] = 0.0;
    body[ibody].vcm[1] = 0.0;
    body[ibody].vcm[2] = 0.0;

    body[ibody].angmom[0] = 0.0;
    body[ibody].angmom[1] = 0.0;
    body[ibody].angmom[2] = 0.0;
  }
}

/* ----------------------------------------------------------------------
   read per rigid body info from user-provided file
   which = 0 to read everything except 6 moments of inertia
   which = 1 to read just 6 moments of inertia
   flag inbody = 0 for local bodies this proc initializes from file
   nlines = # of lines of rigid body info, 0 is OK
   one line = rigid-ID mass xcm ycm zcm ixx iyy izz ixy ixz iyz
              vxcm vycm vzcm lx ly lz
   where rigid-ID = mol-ID for fix rigid/abrade
------------------------------------------------------------------------- */

void FixRigidAbrade::readfile()
{
  int nchunk, eofflag, flag_nlines, atom_nlines, body_nlines, xbox, ybox, zbox;
  FILE *fp;
  char *eof, *start, *next, *buf;
  char line[MAXLINE];
  double temp;

  int nlocal = atom->nlocal;
  int local_id = 0;

  // open file and read atom header and data

  if (me == 0) {
    fp = fopen(inpfile, "r");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix rigid/abrade file {}: {}", inpfile, utils::getsyserror());
    while (true) {
      eof = fgets(line, MAXLINE, fp);
      if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/abrade file");
      start = &line[strspn(line, " \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    flag_nlines = utils::inumeric(FLERR, utils::trim(line), true, lmp);

    // utils::logmesg(lmp, "Reading data for fix rigid/abrade optional keyword flags from file {}\n", inpfile);
    if (flag_nlines == 0) fclose(fp);

  }

  MPI_Bcast(&flag_nlines, 1, MPI_INT, 0, world);

  // empty file with 0 lines is needed to trigger initial restart file
  // generation when no infile was previously used.

  if (flag_nlines == 0)
    return;
  else if (flag_nlines < 0)
    error->all(FLERR, "Fix rigid infile has incorrect format in the flags data section");

  auto *buffer = new char[CHUNK*MAXLINE];
  int nread = 0;
  while (nread < flag_nlines) {
    nchunk = MIN(flag_nlines - nread, CHUNK);
    eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
    if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/abrade file");

    buf = buffer;
    next = strchr(buf,'\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords != 1) // offset_flag -- This can be expanded to included additional flags if desired. If so, make sure changes are reflected in write_restart_file()
      error->all(FLERR,"Incorrect flag format in fix {} file in atom data section", style);

    // loop over lines of attributes
    // tokenize the line into values
    for (int i = 0; i < nchunk; i++) {
      
      next = strchr(buf, '\n');
      *next = '\0';
      
      try {
        ValueTokenizer values(buf);
        int read_offset_flag = values.next_int();

        if (offset_flag != read_offset_flag && me == 0) error->one(FLERR, "fix rigid/abrade: Disagreement between the offset_flag defined in the input script and {}", inpfile);

      } catch (TokenizerException &e) {
        error->all(FLERR, "Invalid fix rigid/abrade infile: {}", e.what());
      }
      buf = next + 1;
    }
    nread += nchunk;
  }


  // read atom header and data

  if (me == 0 && fp) {

    while (true) {
      eof = fgets(line, MAXLINE, fp);
      if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/abrade file");
      start = &line[strspn(line, " \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    atom_nlines = utils::inumeric(FLERR, utils::trim(line), true, lmp);

    utils::logmesg(lmp, "fix rigid/abrade: Reading data for {} bodies from file {}\n", atom_nlines, inpfile);
    if (atom_nlines == 0) fclose(fp);

  }

  MPI_Bcast(&atom_nlines, 1, MPI_INT, 0, world);
  
  // empty file with 0 lines is needed to trigger initial restart file
  // generation when no infile was previously used.

  if (atom_nlines == 0)
    return;
  else if (atom_nlines < 0)
    error->all(FLERR, "Fix rigid infile has incorrect format in the atom data section");

  buffer = new char[CHUNK * MAXLINE];
  nread = 0;
  while (nread < atom_nlines) {
    nchunk = MIN(atom_nlines - nread, CHUNK);
    eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
    if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/abrade file");

    buf = buffer;
    next = strchr(buf,'\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords != 2) // atom ID, cumulative displacement (vertexdata[i][7])
      error->all(FLERR,"Incorrect atom format in fix {} file in atom data section", style);

    // loop over lines of attributes
    // tokenize the line into values
    for (int i = 0; i < nchunk; i++) {
      
      next = strchr(buf, '\n');
      *next = '\0';
      
      try {
        ValueTokenizer values(buf);
        tagint id = values.next_tagint();
        double cumulative_displacement = values.next_double();

        // check if the restart atom is a locally stored atom
        local_id = atom->map(id);
        if (local_id >= 0 && local_id < atom->nlocal){
      
        vertexdata[local_id][7] = cumulative_displacement; 
        }

      } catch (TokenizerException &e) {
        error->all(FLERR, "Invalid fix rigid/abrade infile: {}", e.what());
      }
      buf = next + 1;
    }
    nread += nchunk;
  }


  // read body header and data

  if (me == 0 && fp) {

    while (true) {
      eof = fgets(line, MAXLINE, fp);
      if (eof == nullptr) error->one(FLERR, "Unexpected end of fix rigid/abrade file");
      start = &line[strspn(line, " \t\n\v\f\r")];
      if (*start != '\0' && *start != '#') break;
    }
    body_nlines = utils::inumeric(FLERR, utils::trim(line), true, lmp);

    utils::logmesg(lmp, "fix rigid/abrade: Reading data for {} bodies from file {}\n", body_nlines, inpfile);
    if (body_nlines == 0) fclose(fp);

  }

  MPI_Bcast(&body_nlines, 1, MPI_INT, 0, world);

  // empty file with 0 lines is needed to trigger initial restart file
  // generation when no infile was previously used.

  if (body_nlines == 0)
    return;
  else if (body_nlines < 0)
    error->all(FLERR, "Fix rigid infile has incorrect format in the body data section");

  // auto buffer = new char[CHUNK * MAXLINE];
  nread = 0;
  while (nread < body_nlines) {
    nchunk = MIN(body_nlines - nread, CHUNK);
    eofflag = utils::read_lines_from_file(fp, nchunk, MAXLINE, buffer, me, world);
    if (eofflag) error->all(FLERR, "Unexpected end of fix rigid/abrade file");

    buf = buffer;
    next = strchr(buf,'\n');
    *next = '\0';
    int nwords = utils::count_words(utils::trim_comment(buf));
    *next = '\n';

    if (nwords != RESTART_ATTRIBUTE_PERBODY) // owning atom ID, vcm_x, vcm_y, vcm_z, angmom_x, angmom_y, angmom_z, initial_volume, abraded_volume,  wear_energy
      error->all(FLERR,"Incorrect body format in fix {} file in body data section", style);

    // loop over lines of attributes
    // tokenize the line into values
    for (int i = 0; i < nchunk; i++) {
      
      next = strchr(buf, '\n');
      *next = '\0';
      
      try {
        ValueTokenizer values(buf);
        tagint bodyid = values.next_tagint();
        double vcm_x = values.next_double();
        double vcm_y = values.next_double();
        double vcm_z = values.next_double();
        double angmom_x = values.next_double();
        double angmom_y = values.next_double();
        double angmom_z = values.next_double();
        double initial_volume = values.next_double();
        double abraded_volume = values.next_double();
        double wear_energy = values.next_double();

        // check if the owning restart atom is a locally stored atom and storing body data if so
        local_id = atom->map(bodyid);
        if (local_id >= 0 && local_id < atom->nlocal){
          Body *b = &body[atom2body[local_id]];
          b->vcm[0] = vcm_x;
          b->vcm[1] = vcm_y;
          b->vcm[2] = vcm_z;
          b->angmom[0] = angmom_x;
          b->angmom[1] = angmom_y;
          b->angmom[2] = angmom_z;
          b->initial_volume = initial_volume;
          b->abraded_volume = abraded_volume;
          b->wear_energy = wear_energy;
        }

      } catch (TokenizerException &e) {
        error->all(FLERR, "Invalid fix rigid/abrade infile: {}", e.what());
      }
      buf = next + 1;
    }
    nread += nchunk;
  }

  if (me == 0) fclose(fp);
  delete[] buffer;
}

/* ----------------------------------------------------------------------
   write out restart info for mass, COM, inertia tensor to file
   identical format to inpfile option, so info can be read in when restarting
   each proc contributes info for rigid bodies it owns
------------------------------------------------------------------------- */

void FixRigidAbrade::write_restart_file(const char *file)
{
  //   // forward communicate of vcm to all ghost copies
  // commflag = FULL_BODY;
  // comm->forward_comm(this, 10);
  // commflag = FINAL;
  // comm->forward_comm(this, 10);

  FILE *fp;
  
  // do not write file if bodies have not yet been initialized

  if (!setupflag) return;

  // proc 0 opens file and writes header

  if (me == 0) {
    auto outfile = std::string(file) + ".rigid";
    fp = fopen(outfile.c_str(), "w");
    if (fp == nullptr)
      error->one(FLERR, "Cannot open fix rigid restart file {}: {}", outfile, utils::getsyserror());

  // writing the offset flag used in the previous run
    fmt::print(fp,
               "# fix rigid/abrade offset_flag:\n\n1\n{}\n\n", // Specifying here that there is one row of flags to be read in by readfile(). It would be more elegant to assume there is always one row in readfiles(), but this works. 
               offset_flag); // Additional flags can be added here and then referenced in readfile() if needed.


    // writing the atom section header
    fmt::print(fp,
               "# fix rigid/abrade cumulative data for "
               "{} atoms on timestep {}\n\n",
               atom->natoms, update->ntimestep);
    fmt::print(fp, "{}\n", atom->natoms);
  }

  // --------- --------- Writing the atom data --------- --------- 

  // communication buffer for all my info
  // max_size = largest buffer needed by any proc
  // ncol = # of values per line in output file

  int ncol = 2;
  int sendrow = atom->nlocal;
  int maxrow;
  MPI_Allreduce(&sendrow, &maxrow, 1, MPI_INT, MPI_MAX, world);

  double **buf;
  if (me == 0)
    memory->create(buf, MAX(1, maxrow), ncol, "rigid/abrade:buf");
  else
    memory->create(buf, MAX(1, sendrow), ncol, "rigid/abrade:buf");

  // pack my cumulative data info into buf

  for (int i = 0; i < atom->nlocal; i++) {
    buf[i][0] = atom->tag[i];
    buf[i][1] = vertexdata[i][7]; // cumulative displacement

  }

  // write one chunk of info per proc to file
  // proc 0 pings each proc, receives its chunk, writes to file
  // all other procs wait for ping, send their chunk to proc 0

  int tmp, recvrow;

  if (me == 0) {
    MPI_Status status;
    MPI_Request request;
    for (int iproc = 0; iproc < nprocs; iproc++) {
      if (iproc) {
        MPI_Irecv(&buf[0][0], maxrow * ncol, MPI_DOUBLE, iproc, 0, world, &request);
        MPI_Send(&tmp, 0, MPI_INT, iproc, 0, world);
        MPI_Wait(&request, &status);
        MPI_Get_count(&status, MPI_DOUBLE, &recvrow);
        recvrow /= ncol;
      } else
        recvrow = sendrow;

      for (int i = 0; i < recvrow; i++){
        fprintf(fp,
                "%d %-1.16e\n",
                static_cast<tagint>(buf[i][0]), buf[i][1]);
      }
    }

  } else {
    MPI_Recv(&tmp, 0, MPI_INT, 0, 0, world, MPI_STATUS_IGNORE);
    MPI_Rsend(&buf[0][0], sendrow * ncol, MPI_DOUBLE, 0, 0, world);
  }

  // --------- --------- Writing the body data --------- --------- 

    if (me == 0) {

    // writing the body section header
    fmt::print(fp,
               "\n# fix rigid/abrade cumulative data for "
               "{} bodies on timestep {}\n\n",
               nbody, update->ntimestep);
    fmt::print(fp, "{}\n", nbody);
  }


  // communication buffer for all my info
  // max_size = largest buffer needed by any proc
  // ncol = # of values per line in output file

  ncol = RESTART_ATTRIBUTE_PERBODY;
  sendrow = nlocal_body;
  maxrow = 0;
  MPI_Allreduce(&sendrow, &maxrow, 1, MPI_INT, MPI_MAX, world);

  double **body_buf;
  if (me == 0)
    memory->create(body_buf, MAX(1, maxrow), ncol, "rigid/abrade:body_buf");
  else
    memory->create(body_buf, MAX(1, sendrow), ncol, "rigid/abrade:body_buf");

  // pack my cumulative data info into body_buf

  for (int ibody = 0; ibody < nlocal_body; ibody++) {

    Body *b = &body[ibody];
      body_buf[ibody][0] = atom->tag[b->ilocal]; // owning atom ID
      body_buf[ibody][1] = b->vcm[0]; // vcm_x
      body_buf[ibody][2] = b->vcm[1]; // vcm_y
      body_buf[ibody][3] = b->vcm[2]; // vcm_z
      body_buf[ibody][4] = b->angmom[0]; // angmom_x
      body_buf[ibody][5] = b->angmom[1]; // angmom_y
      body_buf[ibody][6] = b->angmom[2]; // angmom_z
      body_buf[ibody][7] = b->initial_volume; // initial_volume
      body_buf[ibody][8] = b->abraded_volume; // abraded_volume
      body_buf[ibody][9] = b->wear_energy; // wear_energy
  }
  
  // write one chunk of info per proc to file
  // proc 0 pings each proc, receives its chunk, writes to file
  // all other procs wait for ping, send their chunk to proc 0

  if (me == 0) {
    MPI_Status status;
    MPI_Request request;
    for (int iproc = 0; iproc < nprocs; iproc++) {
      if (iproc) {
        MPI_Irecv(&body_buf[0][0], maxrow * ncol, MPI_DOUBLE, iproc, 0, world, &request);
        MPI_Send(&tmp, 0, MPI_INT, iproc, 0, world);
        MPI_Wait(&request, &status);
        MPI_Get_count(&status, MPI_DOUBLE, &recvrow);
        recvrow /= ncol;
      } else
        recvrow = sendrow;

      for (int i = 0; i < recvrow; i++){
        fprintf(fp,
                "%d %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e %-1.16e\n",
                static_cast<tagint>(body_buf[i][0]), body_buf[i][1], body_buf[i][2], body_buf[i][3], body_buf[i][4], body_buf[i][5], body_buf[i][6], body_buf[i][7], body_buf[i][8], body_buf[i][9]);
      }
    }

  } else {
    MPI_Recv(&tmp, 0, MPI_INT, 0, 0, world, MPI_STATUS_IGNORE);
    MPI_Rsend(&body_buf[0][0], sendrow * ncol, MPI_DOUBLE, 0, 0, world);
  }

  // clean up and close file
  memory->destroy(buf);
  memory->destroy(body_buf);
  if (me == 0) fclose(fp);
}



/* ----------------------------------------------------------------------
   randomize rigid body VCMs with respect to a given temperature KT
     and adjust velocities of individual particles accordingly
   mom_flag enforces zeroing of linear momentum for overall system
   called by fix hmc command
------------------------------------------------------------------------- */

void FixRigidAbrade::resample_momenta(int groupbit, int mom_flag, class RanPark *random, double KT)
{
  int nlocal = nlocal_body;
  int ntotal = nghost_body;
  int *mask = atom->mask;

  double stdev, bmass, wbody[3], mbody[3];
  double total_mass = 0;
  Body *b;
  double vcm[] = {0.0, 0.0, 0.0};

  for (int ibody = 0; ibody < nlocal; ibody++) {
    b = &body[ibody];
    if (mask[b->ilocal] & groupbit) {
      bmass = b->mass;
      stdev = sqrt(KT / bmass);
      total_mass += bmass;
      for (int j = 0; j < 3; j++) {
        b->vcm[j] = stdev * random->gaussian();
        vcm[j] += b->vcm[j] * bmass;
        if (b->inertia[j] > 0.0)
          wbody[j] = sqrt(KT / b->inertia[j]) * random->gaussian();
        else
          wbody[j] = 0.0;
      }
    }
    MathExtra::matvec(b->ex_space, b->ey_space, b->ez_space, wbody, b->omega);
  }

  if (mom_flag && (total_mass > 0.0)) {
    for (int j = 0; j < 3; j++) vcm[j] /= total_mass;
    for (int ibody = 0; ibody < nlocal; ibody++) {
      if (mask[b->ilocal] & groupbit) {
        b = &body[ibody];
        for (int j = 0; j < 3; j++) b->vcm[j] -= vcm[j];
      }
    }
  }

  // forward communicate vcm and omega to ghost bodies

  commflag = FINAL;
  comm->forward_comm(this, 10);

  // compute angular momenta of rigid bodies

  for (int ibody = 0; ibody < ntotal; ibody++) {
    b = &body[ibody];
    MathExtra::omega_to_angmom(b->omega, b->ex_space, b->ey_space, b->ez_space, b->inertia,
                               b->angmom);
    MathExtra::transpose_matvec(b->ex_space, b->ey_space, b->ez_space, b->angmom, mbody);
    MathExtra::quatvec(b->quat, mbody, b->conjqm);
    b->conjqm[0] *= 2.0;
    b->conjqm[1] *= 2.0;
    b->conjqm[2] *= 2.0;
    b->conjqm[3] *= 2.0;
  }

  // reset velocities of individual atoms

  set_v();
}


/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixRigidAbrade::grow_arrays(int nmax)
{
  memory->grow(bodyown, nmax, "rigid/abrade:bodyown");
  memory->grow(bodytag, nmax, "rigid/abrade:bodytag");
  memory->grow(atom2body, nmax, "rigid/abrade:atom2body");

  memory->grow(vertexdata, nmax, size_peratom_cols, "rigid/abrade:vertexdata");
  array_atom = vertexdata;

  memory->grow(xcmimage, nmax, "rigid/abrade:xcmimage");
  memory->grow(displace, nmax, 3, "rigid/abrade:displace");  
  memory->grow(unwrap, nmax, 3, "rigid/abrade:unwrap");
  if (extended) {
    memory->grow(eflags, nmax, "rigid/abrade:eflags");
  }

  // check for regrow of vatom
  // must be done whether per-atom virial is accumulated on this step or not
  //   b/c this is only time grow_array() may be called
  // need to regrow b/c vatom is calculated before and after atom migration

  if (nmax > maxvatom) {
    maxvatom = atom->nmax;
    memory->grow(vatom, maxvatom, 6, "fix:vatom");
  }
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixRigidAbrade::copy_arrays(int i, int j, int delflag)
{

  bodytag[j] = bodytag[i];
  xcmimage[j] = xcmimage[i];
  displace[j][0] = displace[i][0];
  displace[j][1] = displace[i][1];
  displace[j][2] = displace[i][2];

  unwrap[j][0] = unwrap[i][0];
  unwrap[j][1] = unwrap[i][1];
  unwrap[j][2] = unwrap[i][2];

  for (int q = 0; q < size_peratom_cols; q++) { vertexdata[j][q] = vertexdata[i][q]; }

  if (extended) {
    eflags[j] = eflags[i];
  }

  // must also copy vatom if per-atom virial calculated on this timestep
  // since vatom is calculated before and after atom migration

  if (vflag_atom)
    for (int k = 0; k < 6; k++) vatom[j][k] = vatom[i][k];

  // if deleting atom J via delflag and J owns a body, then delete it

  if (delflag && bodyown[j] >= 0) {
    bodyown[body[nlocal_body - 1].ilocal] = bodyown[j];
    memcpy(&body[bodyown[j]], &body[nlocal_body - 1], sizeof(Body));
    nlocal_body--;
  }

  // if atom I owns a body, reset I's body.ilocal to loc J
  // do NOT do this if self-copy (I=J) since I's body is already deleted

  if (bodyown[i] >= 0 && i != j) body[bodyown[i]].ilocal = j;
  bodyown[j] = bodyown[i];
}

/* ----------------------------------------------------------------------
   initialize one atom's array values, called when atom is created
------------------------------------------------------------------------- */

void FixRigidAbrade::set_arrays(int i)
{
  bodyown[i] = -1;
  bodytag[i] = 0;
  atom2body[i] = -1;
  xcmimage[i] = 0;
  displace[i][0] = 0.0;
  displace[i][1] = 0.0;
  displace[i][2] = 0.0;

  unwrap[i][0] = 0.0;
  unwrap[i][1] = 0.0;
  unwrap[i][2] = 0.0;

  vertexdata[i][0] = 0.0;
  vertexdata[i][1] = 0.0;
  vertexdata[i][2] = 0.0;
  vertexdata[i][3] = 0.0;
  vertexdata[i][4] = 0.0;
  vertexdata[i][5] = 0.0;
  vertexdata[i][6] = 0.0;
  vertexdata[i][7] = 0.0;

  if (global_normals_flag) {
    vertexdata[i][8] = 0.0;
    vertexdata[i][9] = 0.0;
    vertexdata[i][10] = 0.0;
  }

  // must also zero vatom if per-atom virial calculated on this timestep
  // since vatom is calculated before and after atom migration

  if (vflag_atom)
    for (int k = 0; k < 6; k++) vatom[i][k] = 0.0;
}

/* ----------------------------------------------------------------------
   initialize a molecule inserted by another fix, e.g. deposit or pour
   called when molecule is created
   nlocalprev = # of atoms on this proc before molecule inserted
   tagprev = atom ID previous to new atoms in the molecule
   xgeom = geometric center of new molecule
   vcm = COM velocity of new molecule
   quat = rotation of new molecule (around geometric center)
          relative to template in Molecule class
------------------------------------------------------------------------- */

void FixRigidAbrade::set_molecule(int nlocalprev, tagint tagprev, int imol, double *xgeom,
                                  double *vcm, double *quat)
{
  int m;
  double ctr2com[3], ctr2com_rotate[3];
  double rotmat[3][3];

  // increment total # of rigid bodies

  nbody++;

  // loop over atoms I added for the new body

  int nlocal = atom->nlocal;
  if (nlocalprev == nlocal) return;

  tagint *tag = atom->tag;

  for (int i = nlocalprev; i < nlocal; i++) {
    bodytag[i] = tagprev + onemols[imol]->comatom;
    if (tag[i] - tagprev == onemols[imol]->comatom) bodyown[i] = nlocal_body;

    m = tag[i] - tagprev - 1;
    displace[i][0] = onemols[imol]->dxbody[m][0];
    displace[i][1] = onemols[imol]->dxbody[m][1];
    displace[i][2] = onemols[imol]->dxbody[m][2];

    if (extended) {
      eflags[i] = 0;
      if (onemols[imol]->radiusflag) {
        eflags[i] |= SPHERE;
      }
    }

    if (bodyown[i] >= 0) {
      if (nlocal_body == nmax_body) grow_body();
      Body *b = &body[nlocal_body];
      b->mass = onemols[imol]->masstotal;
      b->natoms = onemols[imol]->natoms;
      b->xgc[0] = xgeom[0];
      b->xgc[1] = xgeom[1];
      b->xgc[2] = xgeom[2];

      // new COM = Q (onemols[imol]->xcm - onemols[imol]->center) + xgeom
      // Q = rotation matrix associated with quat

      MathExtra::quat_to_mat(quat, rotmat);
      MathExtra::sub3(onemols[imol]->com, onemols[imol]->center, ctr2com);
      MathExtra::matvec(rotmat, ctr2com, ctr2com_rotate);
      MathExtra::add3(ctr2com_rotate, xgeom, b->xcm);

      b->vcm[0] = vcm[0];
      b->vcm[1] = vcm[1];
      b->vcm[2] = vcm[2];
      b->inertia[0] = onemols[imol]->inertia[0];
      b->inertia[1] = onemols[imol]->inertia[1];
      b->inertia[2] = onemols[imol]->inertia[2];

      // final quat is product of insertion quat and original quat
      // true even if insertion rotation was not around COM

      MathExtra::quatquat(quat, onemols[imol]->quat, b->quat);
      MathExtra::q_to_exyz(b->quat, b->ex_space, b->ey_space, b->ez_space);

      MathExtra::transpose_matvec(b->ex_space, b->ey_space, b->ez_space, ctr2com_rotate,
                                  b->xgc_body);
      b->xgc_body[0] *= -1;
      b->xgc_body[1] *= -1;
      b->xgc_body[2] *= -1;

      b->angmom[0] = b->angmom[1] = b->angmom[2] = 0.0;
      b->omega[0] = b->omega[1] = b->omega[2] = 0.0;
      b->conjqm[0] = b->conjqm[1] = b->conjqm[2] = b->conjqm[3] = 0.0;

      b->image = ((imageint) IMGMAX << IMG2BITS) | ((imageint) IMGMAX << IMGBITS) | IMGMAX;
      b->ilocal = i;
      nlocal_body++;
    }
  }
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixRigidAbrade::pack_exchange(int i, double *buf)
{
  buf[0] = ubuf(bodytag[i]).d;
  buf[1] = ubuf(xcmimage[i]).d;
  buf[2] = displace[i][0];
  buf[3] = displace[i][1];
  buf[4] = displace[i][2];

  for (int q = 0; q < size_peratom_cols; q++) { buf[5 + q] = vertexdata[i][q]; }

  // extended attribute info
  int m = 5 + size_peratom_cols;
  if (extended) {
    buf[m++] = eflags[i];
  }

  // atom not in a rigid body

  if (!bodytag[i]) return m;

  // must also pack vatom if per-atom virial calculated on this timestep
  // since vatom is calculated before and after atom migration

  if (vflag_atom)
    for (int k = 0; k < 6; k++) buf[m++] = vatom[i][k];

  // atom does not own its rigid body

  if (bodyown[i] < 0) {
    buf[m++] = 0;
    return m;
  }

  // body info for atom that owns a rigid body

  buf[m++] = 1;
  memcpy(&buf[m], &body[bodyown[i]], sizeof(Body));
  m += bodysize;
  return m;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixRigidAbrade::unpack_exchange(int nlocal, double *buf)
{
  bodytag[nlocal] = (tagint) ubuf(buf[0]).i;
  xcmimage[nlocal] = (imageint) ubuf(buf[1]).i;
  displace[nlocal][0] = buf[2];
  displace[nlocal][1] = buf[3];
  displace[nlocal][2] = buf[4];

  for (int q = 0; q < size_peratom_cols; q++) { vertexdata[nlocal][q] = buf[5 + q]; }

  // extended attribute info

  int m = 5 + size_peratom_cols;
  if (extended) {
    eflags[nlocal] = static_cast<int>(buf[m++]);
  }

  // atom not in a rigid body

  if (!bodytag[nlocal]) {
    bodyown[nlocal] = -1;
    return m;
  }

  // must also unpack vatom if per-atom virial calculated on this timestep
  // since vatom is calculated before and after atom migration

  if (vflag_atom)
    for (int k = 0; k < 6; k++) vatom[nlocal][k] = buf[m++];

  // atom does not own its rigid body

  bodyown[nlocal] = static_cast<int>(buf[m++]);
  if (bodyown[nlocal] == 0) {
    bodyown[nlocal] = -1;
    return m;
  }

  // body info for atom that owns a rigid body

  if (nlocal_body == nmax_body) grow_body();
  memcpy(&body[nlocal_body], &buf[m], sizeof(Body));
  m += bodysize;
  body[nlocal_body].ilocal = nlocal;
  bodyown[nlocal] = nlocal_body++;

  return m;
}

/* ----------------------------------------------------------------------
   only pack body info if own or ghost atom owns the body
   for FULL_BODY, send 0/1 flag with every atom
------------------------------------------------------------------------- */

int FixRigidAbrade::pack_forward_comm(int n, int *list, double *buf, int /*pbc_flag*/,
                                      int * /*pbc*/)
{
  int i, j;
  double *xcm, *xgc, *vcm, *quat, *omega, *ex_space, *ey_space, *ez_space, *conjqm;
  double *rmass = atom->rmass;

  int m = 0;

  if (commflag == INITIAL) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      xcm = body[bodyown[j]].xcm;
      buf[m++] = xcm[0];
      buf[m++] = xcm[1];
      buf[m++] = xcm[2];
      xgc = body[bodyown[j]].xgc;
      buf[m++] = xgc[0];
      buf[m++] = xgc[1];
      buf[m++] = xgc[2];
      vcm = body[bodyown[j]].vcm;
      buf[m++] = vcm[0];
      buf[m++] = vcm[1];
      buf[m++] = vcm[2];
      quat = body[bodyown[j]].quat;
      buf[m++] = quat[0];
      buf[m++] = quat[1];
      buf[m++] = quat[2];
      buf[m++] = quat[3];
      omega = body[bodyown[j]].omega;
      buf[m++] = omega[0];
      buf[m++] = omega[1];
      buf[m++] = omega[2];
      ex_space = body[bodyown[j]].ex_space;
      buf[m++] = ex_space[0];
      buf[m++] = ex_space[1];
      buf[m++] = ex_space[2];
      ey_space = body[bodyown[j]].ey_space;
      buf[m++] = ey_space[0];
      buf[m++] = ey_space[1];
      buf[m++] = ey_space[2];
      ez_space = body[bodyown[j]].ez_space;
      buf[m++] = ez_space[0];
      buf[m++] = ez_space[1];
      buf[m++] = ez_space[2];
      conjqm = body[bodyown[j]].conjqm;
      buf[m++] = conjqm[0];
      buf[m++] = conjqm[1];
      buf[m++] = conjqm[2];
      buf[m++] = conjqm[3];
    }

  } else if (commflag == FINAL) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      vcm = body[bodyown[j]].vcm;
      buf[m++] = vcm[0];
      buf[m++] = vcm[1];
      buf[m++] = vcm[2];
      omega = body[bodyown[j]].omega;
      buf[m++] = omega[0];
      buf[m++] = omega[1];
      buf[m++] = omega[2];
      conjqm = body[bodyown[j]].conjqm;
      buf[m++] = conjqm[0];
      buf[m++] = conjqm[1];
      buf[m++] = conjqm[2];
      buf[m++] = conjqm[3];
    }

  } else if (commflag == DISPLACE) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[j] < 0) continue;
      if (!body[atom2body[j]].abraded_flag) continue;

      buf[m++] = displace[j][0];
      buf[m++] = displace[j][1];
      buf[m++] = displace[j][2];
    }

  } else if (commflag == UNWRAP) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[j] < 0) continue;
      if (!body[atom2body[j]].abraded_flag) continue;

      buf[m++] = unwrap[j][0];
      buf[m++] = unwrap[j][1];
      buf[m++] = unwrap[j][2];
    }

  } else if (commflag == DISPLACEMENT_VEL) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[j] < 0) continue;
      if (!body[atom2body[j]].abraded_flag) continue;

      buf[m++] = vertexdata[j][4];
      buf[m++] = vertexdata[j][5];
      buf[m++] = vertexdata[j][6];
    }

  } else if (commflag == NORMALS) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[j] < 0) continue;
      if (!body[atom2body[j]].abraded_flag) continue;

      buf[m++] = vertexdata[j][0];
      buf[m++] = vertexdata[j][1];
      buf[m++] = vertexdata[j][2];
      buf[m++] = vertexdata[j][3];
    }

  } else if (commflag == NEW_ANGLES) {

    // TODO: Can we only cycle through dlist atoms rather than all atoms?

    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to atoms in the dlist
      if (!count(dlist.begin(), dlist.end(), atom->tag[j])) continue;

      // get index of atom in the local dlist to acess its new_angles
      int array_index =
          static_cast<int>(find(dlist.begin(), dlist.end(), atom->tag[j]) - dlist.begin());

      // packing new angles type previous set in the reverse comm of EDGES
      buf[m++] = static_cast<double>(new_angles_type[array_index]);

      // pack and communicate the new_anglws, including the size so that it can be unpacked once sent
      buf[m++] = static_cast<double>(new_angles_list[array_index].size());
      for (int k = 0; k < new_angles_list[array_index].size(); k++) {
        // pack first, second, and third atom in each angle
        buf[m++] = static_cast<double>(new_angles_list[array_index][k][0]);
        buf[m++] = static_cast<double>(new_angles_list[array_index][k][1]);
        buf[m++] = static_cast<double>(new_angles_list[array_index][k][2]);
      }
    }
  } else if (commflag == MASS_NATOMS) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;

      buf[m++] = body[bodyown[j]].mass;
      buf[m++] = static_cast<double>(body[bodyown[j]].natoms);
    }

  } else if (commflag == BODY_MASS) {
    for (i = 0; i < n; i++) {
      j = list[i];

      if (bodyown[j] < 0) continue;
      buf[m++] = body[bodyown[j]].mass;

    }

  } else if (commflag == MIN_AREA) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed FULL_BODY
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;

      buf[m++] = body[bodyown[j]].min_area_atom;
      buf[m++] = static_cast<double>(body[bodyown[j]].min_area_atom_tag);
      buf[m++] = (body[bodyown[j]].surface_area);
      buf[m++] = static_cast<double>(body[bodyown[j]].remesh_atom);
      buf[m++] = static_cast<double>(body[bodyown[j]].natoms);
    }

  } else if (commflag == SURFACE_AREA) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;
      
      buf[m++] = (body[bodyown[j]].surface_area);
    }

  } else if (commflag == EQUALISE) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;

      buf[m++] = equalise_surface_array[bodyown[j]][1];
    }

  } else if (commflag == VCM) {
    for (i = 0; i < n; i++) {
      j = list[i];

      if (bodyown[j] < 0) continue;

      buf[m++] = body[bodyown[j]].vcm[0];
      buf[m++] = body[bodyown[j]].vcm[1];
      buf[m++] = body[bodyown[j]].vcm[2];
    }

  } else if (commflag == ABRADED_FLAG) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;

      buf[m++] = body[bodyown[j]].abraded_flag;
    }

  } else if (commflag == BODYTAG) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = bodytag[j];
    }

  } else if (commflag == FULL_BODY) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0)
        buf[m++] = 0;
      else {
        buf[m++] = 1;
        memcpy(&buf[m], &body[bodyown[j]], sizeof(Body));
        m += bodysize;
      }
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   only ghost atoms are looped over
   for FULL_BODY, store a new ghost body if this atom owns it
   for other commflag values, only unpack body info if atom owns it
------------------------------------------------------------------------- */

void FixRigidAbrade::unpack_forward_comm(int n, int first, double *buf)
{
  int i, j, last;
  double *xcm, *xgc, *vcm, *quat, *omega, *ex_space, *ey_space, *ez_space, *conjqm;
  double *rmass = atom->rmass;

  int m = 0;
  last = first + n;

  if (commflag == INITIAL) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      xcm = body[bodyown[i]].xcm;
      xcm[0] = buf[m++];
      xcm[1] = buf[m++];
      xcm[2] = buf[m++];
      xgc = body[bodyown[i]].xgc;
      xgc[0] = buf[m++];
      xgc[1] = buf[m++];
      xgc[2] = buf[m++];
      vcm = body[bodyown[i]].vcm;
      vcm[0] = buf[m++];
      vcm[1] = buf[m++];
      vcm[2] = buf[m++];
      quat = body[bodyown[i]].quat;
      quat[0] = buf[m++];
      quat[1] = buf[m++];
      quat[2] = buf[m++];
      quat[3] = buf[m++];
      omega = body[bodyown[i]].omega;
      omega[0] = buf[m++];
      omega[1] = buf[m++];
      omega[2] = buf[m++];
      ex_space = body[bodyown[i]].ex_space;
      ex_space[0] = buf[m++];
      ex_space[1] = buf[m++];
      ex_space[2] = buf[m++];
      ey_space = body[bodyown[i]].ey_space;
      ey_space[0] = buf[m++];
      ey_space[1] = buf[m++];
      ey_space[2] = buf[m++];
      ez_space = body[bodyown[i]].ez_space;
      ez_space[0] = buf[m++];
      ez_space[1] = buf[m++];
      ez_space[2] = buf[m++];
      conjqm = body[bodyown[i]].conjqm;
      conjqm[0] = buf[m++];
      conjqm[1] = buf[m++];
      conjqm[2] = buf[m++];
      conjqm[3] = buf[m++];
    }

  } else if (commflag == FINAL) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      vcm = body[bodyown[i]].vcm;
      vcm[0] = buf[m++];
      vcm[1] = buf[m++];
      vcm[2] = buf[m++];
      omega = body[bodyown[i]].omega;
      omega[0] = buf[m++];
      omega[1] = buf[m++];
      omega[2] = buf[m++];
      conjqm = body[bodyown[i]].conjqm;
      conjqm[0] = buf[m++];
      conjqm[1] = buf[m++];
      conjqm[2] = buf[m++];
      conjqm[3] = buf[m++];
    }

  } else if (commflag == DISPLACE) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[i] < 0) continue;
      if (!body[atom2body[i]].abraded_flag) continue;
      displace[i][0] = buf[m++];
      displace[i][1] = buf[m++];
      displace[i][2] = buf[m++];
    }

  } else if (commflag == UNWRAP) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[i] < 0) continue;
      if (!body[atom2body[i]].abraded_flag) continue;
      unwrap[i][0] = buf[m++];
      unwrap[i][1] = buf[m++];
      unwrap[i][2] = buf[m++];
    }

  } else if (commflag == DISPLACEMENT_VEL) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[i] < 0) continue;
      if (!body[atom2body[i]].abraded_flag) continue;
      vertexdata[i][4] = buf[m++];
      vertexdata[i][5] = buf[m++];
      vertexdata[i][6] = buf[m++];
    }

  } else if (commflag == NORMALS) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[i] < 0) continue;
      if (!body[atom2body[i]].abraded_flag) continue;
      vertexdata[i][0] = buf[m++];
      vertexdata[i][1] = buf[m++];
      vertexdata[i][2] = buf[m++];
      vertexdata[i][3] = buf[m++];
    }

  } else if (commflag == NEW_ANGLES) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to atoms in the dlist
      if (!count(dlist.begin(), dlist.end(), atom->tag[i])) continue;

      // get index of atom in the local dlist to acess its stored local new_angles
      int rec_array_index =
          static_cast<int>(find(dlist.begin(), dlist.end(), atom->tag[i]) - dlist.begin());

      int incoming_angle_type = static_cast<int>(buf[m++]);
      new_angles_type[rec_array_index] = incoming_angle_type;

      // unpack the size of the incoming new_angles_list buffer so that the contatined angles can be unpacked
      int rec_angles_size = static_cast<int>(buf[m++]);

      if (rec_angles_size) {

        // unpack new_angles_list
        for (int l = 0; l < rec_angles_size; l++) {
          int angle_atom1 = static_cast<tagint>(buf[m++]);
          int angle_atom2 = static_cast<tagint>(buf[m++]);
          int angle_atom3 = static_cast<tagint>(buf[m++]);

          std::vector<tagint> temp = {angle_atom1, angle_atom2, angle_atom3};
          // check that only one copy is added - required for periodic boundaries
          if (!count(new_angles_list[rec_array_index].begin(),
                     new_angles_list[rec_array_index].end(), temp))
            new_angles_list[rec_array_index].push_back(temp);
        }
      }
    }

  } else if (commflag == MASS_NATOMS) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[i] < 0) continue;
      if (!body[bodyown[i]].abraded_flag) continue;
      
      body[bodyown[i]].mass = buf[m++];
      body[bodyown[i]].natoms = static_cast<int>(buf[m++]);

    }

  } else if (commflag == BODY_MASS) {
    for (i = first; i < last; i++) {

      if (bodyown[i] < 0) continue;
      body[bodyown[i]].mass = buf[m++];

    }

  } else if (commflag == MIN_AREA) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[i] < 0) continue;
      if (!body[bodyown[i]].abraded_flag) continue;

      body[bodyown[i]].min_area_atom = buf[m++];
      body[bodyown[i]].min_area_atom_tag = static_cast<tagint>(buf[m++]);
      body[bodyown[i]].surface_area = (buf[m++]);
      body[bodyown[i]].remesh_atom = static_cast<tagint>(buf[m++]);
      body[bodyown[i]].natoms = static_cast<int>(buf[m++]);
    }

  } else if (commflag == SURFACE_AREA) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[i] < 0) continue;
      if (!body[bodyown[i]].abraded_flag) continue;

      body[bodyown[i]].surface_area = (buf[m++]);
    }

  } else if (commflag == EQUALISE) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[i] < 0) continue;
      if (!body[bodyown[i]].abraded_flag) continue;

      equalise_surface_array[bodyown[i]][1] = buf[m++];
    }

  } else if (commflag == VCM) {
    for (i = first; i < last; i++) {

      if (bodyown[i] < 0) continue;

      body[bodyown[i]].vcm[0] = buf[m++];
      body[bodyown[i]].vcm[1] = buf[m++];
      body[bodyown[i]].vcm[2] = buf[m++];
    }

  } else if (commflag == ABRADED_FLAG) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;

      body[bodyown[i]].abraded_flag = buf[m++];
    }

  } else if (commflag == BODYTAG) {
    for (i = first; i < last; i++) { bodytag[i] = buf[m++]; }

  } else if (commflag == FULL_BODY) {
    for (i = first; i < last; i++) {
      bodyown[i] = static_cast<int>(buf[m++]);
      if (bodyown[i] == 0)
        bodyown[i] = -1;
      else {
        j = nlocal_body + nghost_body;
        if (j == nmax_body) grow_body();
        memcpy(&body[j], &buf[m], sizeof(Body));
        m += bodysize;
        body[j].ilocal = i;
        bodyown[i] = j;
        nghost_body++;
      }
    }
  }
}

/* ----------------------------------------------------------------------
   only ghost atoms are looped over
   only pack body info if atom owns it
------------------------------------------------------------------------- */

int FixRigidAbrade::pack_reverse_comm(int n, int first, double *buf)
{
  int i, j, m, last;
  double *fcm, *torque, *vcm, *angmom, *xcm, *xgc;

  m = 0;
  last = first + n;

  if (commflag == FORCE_TORQUE) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      fcm = body[bodyown[i]].fcm;
      buf[m++] = fcm[0];
      buf[m++] = fcm[1];
      buf[m++] = fcm[2];
      torque = body[bodyown[i]].torque;
      buf[m++] = torque[0];
      buf[m++] = torque[1];
      buf[m++] = torque[2];
    }

  } else if (commflag == VCM) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      vcm = body[bodyown[i]].vcm;
      buf[m++] = vcm[0];
      buf[m++] = vcm[1];
      buf[m++] = vcm[2];
    }

  } else if (commflag == ANGMOM) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      angmom = body[bodyown[i]].angmom;
      buf[m++] = angmom[0];
      buf[m++] = angmom[1];
      buf[m++] = angmom[2];
    }

  } else if (commflag == XCM_MASS) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[i] < 0) continue;
      if (!body[bodyown[i]].abraded_flag) continue;

      xcm = body[bodyown[i]].xcm;
      xgc = body[bodyown[i]].xgc;
      buf[m++] = xcm[0];
      buf[m++] = xcm[1];
      buf[m++] = xcm[2];
      buf[m++] = xgc[0];
      buf[m++] = xgc[1];
      buf[m++] = xgc[2];
      buf[m++] = body[bodyown[i]].mass;
      buf[m++] = body[bodyown[i]].volume;
      buf[m++] = static_cast<double>(body[bodyown[i]].natoms);
    }

  } else if (commflag == MIN_AREA) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[i] < 0) continue;
      if (!body[bodyown[i]].abraded_flag) continue;

      buf[m++] = body[bodyown[i]].min_area_atom;
      buf[m++] = static_cast<double>(body[bodyown[i]].min_area_atom_tag);
      buf[m++] = body[bodyown[i]].surface_area;
      buf[m++] = static_cast<double>(body[bodyown[i]].remesh_atom);
      buf[m++] = body[bodyown[i]].remesh_atom_displacement_vel;
    }

  } else if (commflag == SURFACE_AREA) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[i] < 0) continue;
      if (!body[bodyown[i]].abraded_flag) continue;

      buf[m++] = body[bodyown[i]].surface_area;
    }

  } else if (commflag == WEAR_ENERGY) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[i] < 0) continue;
      if (!body[bodyown[i]].abraded_flag) continue;

      buf[m++] = body[bodyown[i]].wear_energy;
    }

  } else if (commflag == EQUALISE) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies
      if (bodyown[i] < 0) continue;

      buf[m++] = equalise_surface_array[bodyown[i]][1];
    }

  } else if (commflag == ABRADED_FLAG) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      buf[m++] = body[bodyown[i]].abraded_flag;
    }
  } else if (commflag == ITENSOR) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;

      if (!body[bodyown[i]].abraded_flag) continue;

      j = bodyown[i];
      buf[m++] = itensor[j][0];
      buf[m++] = itensor[j][1];
      buf[m++] = itensor[j][2];
      buf[m++] = itensor[j][3];
      buf[m++] = itensor[j][4];
      buf[m++] = itensor[j][5];
    }

  } else if (commflag == NORMALS) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[i] < 0) continue;
      if (!body[atom2body[i]].abraded_flag) continue;

      buf[m++] = vertexdata[i][0];
      buf[m++] = vertexdata[i][1];
      buf[m++] = vertexdata[i][2];
      buf[m++] = vertexdata[i][3];
    }

  } else if (commflag == EDGES) {
    for (i = first; i < last; i++) {

      // Only communicating properties relevant to atoms in the dlist
      if (!count(dlist.begin(), dlist.end(), atom->tag[i])) continue;

      // get index of atom in the local dlist to acess its stored local edges
      int array_index =
          static_cast<int>(find(dlist.begin(), dlist.end(), atom->tag[i]) - dlist.begin());

      // packing new angles type
      buf[m++] = static_cast<double>(new_angles_type[array_index]);

      // pack and communicate the associated edges, including the size so that it can be unpacked once sent
      buf[m++] = static_cast<double>(edges[array_index].size());
      for (int k = 0; k < edges[array_index].size(); k++) {
        // pack first and second atom in each edge
        buf[m++] = static_cast<double>(edges[array_index][k][0]);
        buf[m++] = static_cast<double>(edges[array_index][k][1]);
      }

      // clear the edges after they are packed so that a boundary is not constructed on a processor which does not own the remeshing atom
      edges[array_index].clear();
    }

  } else if (commflag == DOF) {
    for (i = first; i < last; i++) {
      if (bodyown[i] < 0) continue;
      j = bodyown[i];
      buf[m++] = counts[j][0];
      buf[m++] = counts[j][1];
      buf[m++] = counts[j][2];
    }
  }

  return m;
}

/* ----------------------------------------------------------------------
   only unpack body info if own or ghost atom owns the body
------------------------------------------------------------------------- */

void FixRigidAbrade::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i, j, k;
  double *fcm, *torque, *vcm, *angmom, *xcm, *xgc;

  int m = 0;

  if (commflag == FORCE_TORQUE) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      fcm = body[bodyown[j]].fcm;
      fcm[0] += buf[m++];
      fcm[1] += buf[m++];
      fcm[2] += buf[m++];
      torque = body[bodyown[j]].torque;
      torque[0] += buf[m++];
      torque[1] += buf[m++];
      torque[2] += buf[m++];
    }

  } else if (commflag == VCM) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      vcm = body[bodyown[j]].vcm;
      vcm[0] += buf[m++];
      vcm[1] += buf[m++];
      vcm[2] += buf[m++];
    }

  } else if (commflag == ANGMOM) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      angmom = body[bodyown[j]].angmom;
      angmom[0] += buf[m++];
      angmom[1] += buf[m++];
      angmom[2] += buf[m++];
    }

  } else if (commflag == XCM_MASS) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;

      xcm = body[bodyown[j]].xcm;
      xgc = body[bodyown[j]].xgc;
      xcm[0] += buf[m++];
      xcm[1] += buf[m++];
      xcm[2] += buf[m++];
      xgc[0] += buf[m++];
      xgc[1] += buf[m++];
      xgc[2] += buf[m++];
      body[bodyown[j]].mass += buf[m++];
      body[bodyown[j]].volume += buf[m++];
      body[bodyown[j]].natoms += static_cast<int>(buf[m++]);
    }

  } else if (commflag == MIN_AREA) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;

      // check that the incoming buffer (containing the minimum associated area of the ghost body) is smaller than the locally stored value
      // Additionally, check that its atom tag is larger (to create consistency across multiple runs if two incoming atoms have the same area)

      double incoming_min_area_atom = buf[m++];
      int incoming_min_area_atom_tag = static_cast<tagint>(buf[m++]);

      if ((incoming_min_area_atom <= (body[bodyown[j]].min_area_atom * (1.0 + 0.001)))) {

        // if the incoming area is equal to the current stored area then give precident to the larger atom tag
        if (incoming_min_area_atom < (body[bodyown[j]].min_area_atom * (1.0 - 0.001))) {
          body[bodyown[j]].min_area_atom = incoming_min_area_atom;
          body[bodyown[j]].min_area_atom_tag = incoming_min_area_atom_tag;
        } else {
          if (incoming_min_area_atom_tag > body[bodyown[j]].min_area_atom_tag) {

            body[bodyown[j]].min_area_atom = incoming_min_area_atom;
            body[bodyown[j]].min_area_atom_tag = incoming_min_area_atom_tag;
          }
        }
      }

      body[bodyown[j]].surface_area += buf[m++];

      // check if an incoming remesh atom has a largest displacement velocity than the local remeshing atom (will be 0 if there is no local remeshing atom)
      // This creates consistency when remeshing across multiple procs
      int incoming_remesh_atom = static_cast<int>(buf[m++]);
      double incoming_remesh_atom_displacement_vel = buf[m++];

      if (incoming_remesh_atom_displacement_vel > body[bodyown[j]].remesh_atom_displacement_vel) { 
        body[bodyown[j]].remesh_atom = incoming_remesh_atom;
        body[bodyown[j]].remesh_atom_displacement_vel = incoming_remesh_atom_displacement_vel;
      }    

    }

  } else if (commflag == SURFACE_AREA) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;

      body[bodyown[j]].surface_area += buf[m++];
    }

  } else if (commflag == WEAR_ENERGY) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;

      body[bodyown[j]].wear_energy += buf[m++];
    }

  } else if (commflag == EQUALISE) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies
      if (bodyown[j] < 0) continue;

      equalise_surface_array[bodyown[j]][1] += (buf[m++]);
    }

  } else if (commflag == ABRADED_FLAG) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      body[bodyown[j]].abraded_flag += buf[m++];
    }
  } else if (commflag == ITENSOR) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (bodyown[j] < 0) continue;
      if (!body[bodyown[j]].abraded_flag) continue;

      k = bodyown[j];
      itensor[k][0] += buf[m++];
      itensor[k][1] += buf[m++];
      itensor[k][2] += buf[m++];
      itensor[k][3] += buf[m++];
      itensor[k][4] += buf[m++];
      itensor[k][5] += buf[m++];
    }

  } else if (commflag == NORMALS) {
    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to bodies which have abraded and changed shape
      if (atom2body[j] < 0) continue;
      if (!body[atom2body[j]].abraded_flag) continue;

      vertexdata[j][0] += buf[m++];
      vertexdata[j][1] += buf[m++];
      vertexdata[j][2] += buf[m++];
      vertexdata[j][3] += buf[m++];
    }

  } else if (commflag == EDGES) {

    for (i = 0; i < n; i++) {
      j = list[i];

      // Only communicating properties relevant to atoms in the dlist
      if (!count(dlist.begin(), dlist.end(), atom->tag[j])) continue;

      // get index of atom in the local dlist to acess its stored local edges
      int rec_array_index =
          static_cast<tagint>(find(dlist.begin(), dlist.end(), atom->tag[j]) - dlist.begin());

      // unpacking new angles type if it has been set
      int incoming_angle_type = static_cast<int>(buf[m++]);
      if (incoming_angle_type != -1) new_angles_type[rec_array_index] = incoming_angle_type;

      // unpack the size of the incoming edges buffer so that the contatined edges can be unpacked
      int rec_edges_size = static_cast<int>(buf[m++]);

      if (rec_edges_size) {
        for (int l = 0; l < rec_edges_size; l++) {
          int edge_atom1 = static_cast<tagint>(buf[m++]);
          int edge_atom2 = static_cast<tagint>(buf[m++]);
          edges[rec_array_index].push_back({edge_atom1, edge_atom2});
        }
      }
    }

  } else if (commflag == DOF) {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (bodyown[j] < 0) continue;
      k = bodyown[j];
      counts[k][0] += static_cast<int>(buf[m++]);
      counts[k][1] += static_cast<int>(buf[m++]);
      counts[k][2] += static_cast<int>(buf[m++]);
    }
  }
}

/* ----------------------------------------------------------------------
   grow body data structure
------------------------------------------------------------------------- */

void FixRigidAbrade::grow_body()
{
  nmax_body += DELTA_BODY;
  body = (Body *) memory->srealloc(body, nmax_body * sizeof(Body), "rigid/abrade:body");
}

/* ----------------------------------------------------------------------
   reset atom2body for all owned atoms
   do this via bodyown of atom that owns the body the owned atom is in
   atom2body values can point to original body or any image of the body
------------------------------------------------------------------------- */

void FixRigidAbrade::reset_atom2body()
{

  int iowner;

  // forward communicate bodytag[i] for ghost atoms
  commflag = BODYTAG;
  comm->forward_comm(this, 1);

  // iowner = index of atom that owns the body that atom I is in
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;

  for (int i = 0; i < (nlocal + nghost); i++) {
    atom2body[i] = -1;
    if (bodytag[i]) {
      iowner = atom->map(bodytag[i]);
      if (iowner == -1)
        error->one(FLERR,
                   "Rigid body atoms {} {} missing on "
                   "proc {} at step {}",
                   atom->tag[i], bodytag[i], comm->me, update->ntimestep);

      atom2body[i] = bodyown[iowner];
    }
  }
}

/* ---------------------------------------------------------------------- */

void FixRigidAbrade::reset_dt()
{
  dtv = update->dt;
  dtf = 0.5 * update->dt * force->ftm2v;
  dtq = 0.5 * update->dt;
}

/* ----------------------------------------------------------------------
   zero linear momentum of each rigid body
   set Vcm to 0.0, then reset velocities of particles via set_v()
------------------------------------------------------------------------- */

void FixRigidAbrade::zero_momentum()
{
  double *vcm;
  for (int ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    vcm = body[ibody].vcm;
    vcm[0] = vcm[1] = vcm[2] = 0.0;
  }

  // forward communicate of vcm to all ghost copies

  commflag = FINAL;
  comm->forward_comm(this, 10);

  // set velocity of atoms in rigid bodues

  evflag = 0;
  set_v();
}

/* ----------------------------------------------------------------------
   zero angular momentum of each rigid body
   set angmom/omega to 0.0, then reset velocities of particles via set_v()
------------------------------------------------------------------------- */

void FixRigidAbrade::zero_rotation()
{
  double *angmom, *omega;
  for (int ibody = 0; ibody < nlocal_body + nghost_body; ibody++) {
    angmom = body[ibody].angmom;
    angmom[0] = angmom[1] = angmom[2] = 0.0;
    omega = body[ibody].omega;
    omega[0] = omega[1] = omega[2] = 0.0;
  }

  // forward communicate of omega to all ghost copies

  commflag = FINAL;
  comm->forward_comm(this, 10);

  // set velocity of atoms in rigid bodues

  evflag = 0;
  set_v();
}

/* ---------------------------------------------------------------------- */

int FixRigidAbrade::modify_param(int narg, char **arg)
{
  if (strcmp(arg[0], "bodyforces") == 0) {
    if (narg < 2) error->all(FLERR, "Illegal fix_modify command");
    if (strcmp(arg[1], "early") == 0)
      earlyflag = 1;
    else if (strcmp(arg[1], "late") == 0)
      earlyflag = 0;
    else
      error->all(FLERR, "Illegal fix_modify command");

    // reset fix mask
    // must do here and not in init,
    // since modify.cpp::init() uses fix masks before calling fix::init()

    for (int i = 0; i < modify->nfix; i++)
      if (strcmp(modify->fix[i]->id, id) == 0) {
        if (earlyflag)
          modify->fmask[i] |= POST_FORCE;
        else if (!langflag)
          modify->fmask[i] &= ~POST_FORCE;
        break;
      }

    return 2;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

void *FixRigidAbrade::extract(const char *str, int &dim)
{
  dim = 0;

  if (strcmp(str, "body") == 0) {
    if (!setupflag) return nullptr;
    dim = 1;
    return atom2body;
  }

  if (strcmp(str, "onemol") == 0) {
    dim = 0;
    return onemols;
  }

  // return vector of rigid body masses, for owned+ghost bodies
  // used by granular pair styles, indexed by atom2body

  if (strcmp(str, "masstotal") == 0) {

    commflag = BODY_MASS;
    comm->forward_comm(this, 1);

    if (!setupflag) return nullptr;
    dim = 1;

    if (nmax_mass < nmax_body) {
      memory->destroy(mass_body);
      nmax_mass = nmax_body;
      memory->create(mass_body, nmax_mass, "rigid:mass_body");
    }

    int n = nlocal_body + nghost_body;
    for (int i = 0; i < n; i++) mass_body[i] = body[i].mass;

    return mass_body;
  }

  return nullptr;
}

/* ----------------------------------------------------------------------
   return translational KE for all rigid bodies
   KE = 1/2 M Vcm^2
   sum local body results across procs
------------------------------------------------------------------------- */

double FixRigidAbrade::extract_ke()
{
  double *vcm;

  double ke = 0.0;
  for (int i = 0; i < nlocal_body; i++) {
    vcm = body[i].vcm;
    ke += body[i].mass * (vcm[0] * vcm[0] + vcm[1] * vcm[1] + vcm[2] * vcm[2]);
  }

  double keall;
  MPI_Allreduce(&ke, &keall, 1, MPI_DOUBLE, MPI_SUM, world);

  return 0.5 * keall;
}

/* ----------------------------------------------------------------------
   return rotational KE for all rigid bodies
   Erotational = 1/2 I wbody^2
------------------------------------------------------------------------- */

double FixRigidAbrade::extract_erotational()
{
  double wbody[3], rot[3][3];
  double *inertia;

  double erotate = 0.0;
  for (int i = 0; i < nlocal_body; i++) {

    // for Iw^2 rotational term, need wbody = angular velocity in body frame
    // not omega = angular velocity in space frame

    inertia = body[i].inertia;
    MathExtra::quat_to_mat(body[i].quat, rot);
    MathExtra::transpose_matvec(rot, body[i].angmom, wbody);
    if (inertia[0] == 0.0)
      wbody[0] = 0.0;
    else
      wbody[0] /= inertia[0];
    if (inertia[1] == 0.0)
      wbody[1] = 0.0;
    else
      wbody[1] /= inertia[1];
    if (inertia[2] == 0.0)
      wbody[2] = 0.0;
    else
      wbody[2] /= inertia[2];

    erotate += inertia[0] * wbody[0] * wbody[0] + inertia[1] * wbody[1] * wbody[1] +
        inertia[2] * wbody[2] * wbody[2];
  }

  double erotateall;
  MPI_Allreduce(&erotate, &erotateall, 1, MPI_DOUBLE, MPI_SUM, world);

  return 0.5 * erotateall;
}

/* ----------------------------------------------------------------------
   return temperature of collection of rigid bodies
   non-active DOF are removed by fflag/tflag and in tfactor
------------------------------------------------------------------------- */

double FixRigidAbrade::compute_scalar()
{
  double wbody[3], rot[3][3];

  double *vcm, *inertia;

  double t = 0.0;

  for (int i = 0; i < nlocal_body; i++) {
    vcm = body[i].vcm;
    t += body[i].mass * (vcm[0] * vcm[0] + vcm[1] * vcm[1] + vcm[2] * vcm[2]);

    // for Iw^2 rotational term, need wbody = angular velocity in body frame
    // not omega = angular velocity in space frame

    inertia = body[i].inertia;
    MathExtra::quat_to_mat(body[i].quat, rot);
    MathExtra::transpose_matvec(rot, body[i].angmom, wbody);
    if (inertia[0] == 0.0)
      wbody[0] = 0.0;
    else
      wbody[0] /= inertia[0];
    if (inertia[1] == 0.0)
      wbody[1] = 0.0;
    else
      wbody[1] /= inertia[1];
    if (inertia[2] == 0.0)
      wbody[2] = 0.0;
    else
      wbody[2] /= inertia[2];

    t += inertia[0] * wbody[0] * wbody[0] + inertia[1] * wbody[1] * wbody[1] +
        inertia[2] * wbody[2] * wbody[2];
  }

  double tall;
  MPI_Allreduce(&t, &tall, 1, MPI_DOUBLE, MPI_SUM, world);

  double tfactor = force->mvv2e / ((6.0 * nbody - nlinear) * force->boltz);
  tall *= tfactor;
  return tall;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays
------------------------------------------------------------------------- */

double FixRigidAbrade::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = (double) nmax * 2 * sizeof(int);
  bytes += (double) nmax * sizeof(imageint);
  bytes += (double) nmax * 3 * sizeof(double);
  bytes += (double) maxvatom * 6 * sizeof(double);    // vatom
  if (extended) {
    bytes += (double) nmax * sizeof(int);
  }
  bytes += (double) nmax_body * sizeof(Body);

  bytes += atom->nmax * size_peratom_cols * sizeof(double);    //~ For vertexdata array
  return bytes;
}

/* ----------------------------------------------------------------------
   debug method for sanity checking of atom/body data pointers
------------------------------------------------------------------------- */

/*
void FixRigidAbrade::check(int flag)
{
  for (int i = 0; i < atom->nlocal; i++) {
    if (bodyown[i] >= 0) {
      if (bodytag[i] != atom->tag[i]) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD AAA");
      }
      if (bodyown[i] < 0 || bodyown[i] >= nlocal_body) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD BBB");
      }
      if (atom2body[i] != bodyown[i]) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD CCC");
      }
      if (body[bodyown[i]].ilocal != i) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD DDD");
      }
    }
  }

  for (int i = 0; i < atom->nlocal; i++) {
    if (bodyown[i] < 0 && bodytag[i] > 0) {
      if (atom2body[i] < 0 || atom2body[i] >= nlocal_body+nghost_body) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD EEE");
      }
      if (bodytag[i] != atom->tag[body[atom2body[i]].ilocal]) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD FFF");
      }
    }
  }

  for (int i = atom->nlocal; i < atom->nlocal + atom->nghost; i++) {
    if (bodyown[i] >= 0) {
      if (bodyown[i] < nlocal_body ||
          bodyown[i] >= nlocal_body+nghost_body) {
        printf("Values %d %d: %d %d %d\n",
               i,atom->tag[i],bodyown[i],nlocal_body,nghost_body);
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD GGG");
      }
      if (body[bodyown[i]].ilocal != i) {
        printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
        errorx->one(FLERR,"BAD HHH");
      }
    }
  }

  for (int i = 0; i < nlocal_body; i++) {
    if (body[i].ilocal < 0 || body[i].ilocal >= atom->nlocal) {
      printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
      errorx->one(FLERR,"BAD III");
    }
    if (bodytag[body[i].ilocal] != atom->tag[body[i].ilocal] ||
        bodyown[body[i].ilocal] != i) {
      printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
      errorx->one(FLERR,"BAD JJJ");
    }
  }

  for (int i = nlocal_body; i < nlocal_body + nghost_body; i++) {
    if (body[i].ilocal < atom->nlocal ||
        body[i].ilocal >= atom->nlocal + atom->nghost) {
      printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
      errorx->one(FLERR,"BAD KKK");
    }
    if (bodyown[body[i].ilocal] != i) {
      printf("Proc %d, step %ld, flag %d\n",comm->me,update->ntimestep,flag);
      errorx->one(FLERR,"BAD LLL");
    }
  }
}
*/
