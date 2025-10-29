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

#include "fix_limit.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "input.h"
#include "math_const.h"
#include "math_extra.h"
#include "memory.h"
#include "modify.h"
#include "respa.h"
#include "random_mars.h"
#include "update.h"
#include "variable.h"

#include <cmath>
#include <cstring>
#include <random>


using namespace LAMMPS_NS;
using namespace FixConst;
using namespace MathConst;

enum{CHUTE,SPHERICAL,VECTOR};

/* ---------------------------------------------------------------------- */

FixLimit::FixLimit(LAMMPS *_lmp, int narg, char **arg) :
  Fix(_lmp, narg, arg),
  mstr(nullptr), vstr(nullptr), pstr(nullptr), tstr(nullptr),
  xstr(nullptr), ystr(nullptr), zstr(nullptr)
{

  if (narg < 1) error->all(FLERR,"Illegal fix limit command");

   upper_v = utils::numeric(FLERR,arg[3],false,lmp);
  
  
  dynamic_group_allow = 1;
  scalar_flag = 1;
  global_freq = 1;
  extscalar = 1;
  energy_global_flag = 1;
  respa_level_support = 1;
  ilevel_respa = 0;

  atom->add_callback(Atom::GROW);
  atom->add_callback(Atom::RESTART);
}

/* ---------------------------------------------------------------------- */

FixLimit::~FixLimit()
{
  if (copymode) return;
  delete [] mstr;
  delete [] vstr;
  delete [] pstr;
  delete [] tstr;
  delete [] xstr;
  delete [] ystr;
  delete [] zstr;
}

/* ---------------------------------------------------------------------- */

int FixLimit::setmask()
{
  int mask = 0;
  mask |= END_OF_STEP;
  return mask;
}

/* ----------------------------------------------------------------------
   END OF STEP
------------------------------------------------------------------------- */
void FixLimit::end_of_step()
{

 int *mask = atom->mask;
 int nlocal = atom->nlocal;
 double **v = atom->v;

 for (int i = 0; i < nlocal; i++){
    if (!(mask[i] & groupbit)) continue;
        
      if (MathExtra::len3(v[i]) > upper_v) {
         v[i][0] = v[i][0] / (MathExtra::len3(v[i])/upper_v);
         v[i][1] = v[i][1] / (MathExtra::len3(v[i])/upper_v);
         v[i][2] = v[i][2] / (MathExtra::len3(v[i])/upper_v);        
      }
 }
   
}

