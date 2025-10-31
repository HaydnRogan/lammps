.. index:: fix limit

fix limit command
=================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID limit upper_v

* ID, group-ID are documented in :doc:`fix <fix>` command
* limit = style name of this fix command
* upper_v = upper velocity magnitude 

Examples
""""""""

.. code-block:: LAMMPS
   
   fix 2 projectile limit 1000

   variable v_max equal 500
   fix 3 projectile limit ${v_max}
   
Description
"""""""""""

At the end of each timestep, the targeted atom's velocity magnitudes are compared against *upper_v*. If an 
atom's magnitude exceeds *upper_v*, then the x, y, z components of the atom's velocity are scaled by the ratio 
of *upper_v* to the atom's velocity magnitude. In this way, the direction of travel for the atom is preserved, 
but the magnitude of velocity does not exceed *upper_v*.

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
No information about this fix is written to :doc:`binary restart files <restart>`.

No parameter of this fix can be used with the *start/stop* keywords of
the :doc:`run <run>` command.  This fix is not invoked during
:doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the RIGID package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

none

Default
"""""""

none
