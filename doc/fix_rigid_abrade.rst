.. index:: fix rigid/abrade

fix rigid/abrade command
========================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID rigid/abrade normal_hardness hardness_ratio density molecule keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* rigid/abrade = style name of this fix command
* normal_hardness = particle's resistance to indentation (pressure units)
* hardness_ratio = multiplier to calculate the tangential_hardness representing the particle's resistance to scratching 
* density = mass density of the particle bounded by its triangulated surface (mass density units)
* molecule = bodystyle supported by this fix command

* zero or more keyword/value pairs may be appended
* keyword = *normals* or *remesh* or *equalise* or *offset* or *stationary* or *langevin* or *reinit* or *temp* or *iso* or *aniso* or *x* or *y* or *z* or *couple* or *tparam* or *pchain* or *dilate* or *infile* or *gravity*

  .. parsed-literal::

       *normals* value = none
       *remesh* value = coeff
         coeff = multiplier between 0 and 1 determining the onset of re-meshing
       *equalise* value = none
       *offset* value = none
       *stationary* value = none
       *langevin* values = Tstart Tstop Tperiod seed
         Tstart,Tstop = desired temperature at start/stop of run (temperature units)
         Tdamp = temperature damping parameter (time units)
         seed = random number seed to use for white noise (positive integer)
       *reinit* value = *yes* or *no*
       *temp* values = Tstart Tstop Tdamp
         Tstart,Tstop = desired temperature at start/stop of run (temperature units)
         Tdamp = temperature damping parameter (time units)
       *iso* or *aniso* values = Pstart Pstop Pdamp
         Pstart,Pstop = scalar external pressure at start/end of run (pressure units)
         Pdamp = pressure damping parameter (time units)
       *x* or *y* or *z* values = Pstart Pstop Pdamp
         Pstart,Pstop = external stress tensor component at start/end of run (pressure units)
         Pdamp = stress damping parameter (time units)
       *couple* value = *none* or *xyz* or *xy* or *yz* or *xz*
       *tparam* values = Tchain Titer Torder
         Tchain = length of Nose/Hoover thermostat chain
         Titer = number of thermostat iterations performed
         Torder = 3 or 5 = Yoshida-Suzuki integration parameters
       *pchain* values = Pchain
         Pchain = length of the Nose/Hoover thermostat chain coupled with the barostat
       *dilate* value = dilate-group-ID
         dilate-group-ID = only dilate atoms in this group due to barostat volume changes
       *infile* filename
         filename = file with per-body values of mass, center of mass, moments of inertia
       *gravity* values = gravity-ID
         gravity-ID = ID of fix gravity command to add gravitational forces


Examples
""""""""

.. code-block:: LAMMPS
   
   fix 1 particles rigid/abrade 1e10 0.25 8 molecule 
   fix 1 particles rigid/abrade 1e20 0.4 10 molecule normals stationary   
   fix 1 all rigid/abrade ${H} 0.25 8 molecule normals gravity 1 
   fix 1 all rigid/abrade ${H} 0.25 ${density} molecule remesh 0.95 equalise  
   fix 1 all rigid/abrade ${H} 0.25 ${density} molecule offset langevin 5e19 1e20 1e-2 1111 

   fix 1 all rigid/abrade 1e9 0.25 1.42 molecule remesh 0.95 equalise normals offset
   
   read_restart example_restart.100
   fix 1 all rigid/abrade 1e9 0.25 1.42 molecule remesh 0.95 equalise normals offset infile example_restart.100.rigid

Description
"""""""""""


As an extension of the :doc:`fix rigid/small <fix_rigid>` style, the *rigid/abrade* style treats triangulated hollow closed surfaces, comprised of discrete spherical atoms, 
as independent abradable bodies. The spherical atoms describing the surface form the nodes of a triangulated mesh, providing well-defined local areas
and normals. Following an impact exceeding a material hardness :ref:`(Capozza) <Capozza>`, spheres are displaced inwards along these normals. The
result is a reduction in volume and a permanent change in particle shape. Each abraded particle’s moment of inertia is then recomputed through the triangulation and used to
resolve future rigid-body dynamics. In this way, particle-level changes in shape are communicated up to the system’s bulk dynamics, which in turn informs
subsequent abrasion. Unless otherwise stated, the guidance, restrictions, and underlying rigid-body functionality are unchanged from the *rigid/small* style. The
listed optional keywords also largely comprise those supported by the *rigid/small* style. Detailed descriptions are reserved for features which deviate from the original *rigid/small* style.

.. warning::

   The :doc:`newton <newton>` on command is required for the *rigid/abrade* style to be used. 

Fix *rigid/abrade* supports the molecule bodystyle. Abradable particles are defined through the :doc:`molecule <molecule>` command and corresponding 
molecule file. This file contains the spherical surface atoms' positions and diameters. A series of :doc:`angles <angles>` are also outlined to store the 
triangulation of the closed surface. Lastly, an additional atom is placed close to the expected center of mass of the abradable particle. This atom 
will be assigned as the owning atom, store the respective body's information, and will be relocated to the calculated center of mass throughout the simulation. 

.. warning::

   If the owning atom lies outside of the particle's surface, e.g., for a torus, then its pairwise interactions should be disabled.

.. note::

   The common STL file format can be translated to provide the triangulation information for the molecule file. The surface atom radius should then be set to ensure sufficient overlap to avoid voids in the surface.  

.. note::

   The *rigid/abrade* style requires the use of hybrid neighbor lists. The ghost atom cutoff for the owning atom, which is repeatedly placed at the center of mass of the body, must be large
   enough to span the ghost atom cutoff of every other atom in the body. This can be achieved through the input script by defining a group containing all owning atoms and setting up the neighbor
   lists accordingly through the :doc:`neigh_modify <neigh_modify>` and :doc:`comm_modify <comm_modify>` commands. 


Rigid-body properties are calculated through the triangulation defined about the surface atom's positions. Each facet is 
stored in the :doc:`angles <angles>` data structure and connected to a common origin to form a series of tetrahedra. From these tetrahedra, the volume, mass, and 
inertia of the filled particles are calculated :ref:`(Tonon) <Tonon>` and used to resolve the rigid-body dynamics.

.. note::

   The :doc:`angles <angles>` are only used to store the triangulation through groupings of surface atoms. Interactions for angles should be disabled through the :doc:`angle_style <angle_style>` *none* command.

To abrade the surface of a particle, pairwise forces are computed for each surface atom and summed to the body as in the 
*rigid/small* style. For *rigid/abrade*, these forces are further decomposed into normal and shear stresses by considering 
their associated surface normals and areas calculated through the triangulation. The imposed normal stress is compared against
the *normal_hardness*, and the shear stress is compared against the product of the *normal_hardness* and 
*hardness_ratio*. If either hardness is exceeded, then abrasion is initiated and an abrasion velocity is calculated for the respective surface atom :ref:`(Capozza) <Capozza>`. Surface atoms are integrated by their abrasion velocity inwards along their associated normals in their body coordinate systems. 
This permanently alters the particle shape and triangulation. Following any change in shape, the rigid-body properties for abraded particles are recomputed with respect to their updated triangulations. 

.. note::

   The *rigid/abrade* style also supports abrasion with surfaces defined through *fix wall/gran/region* available from the GRANULAR package.

By default, the surface normals are calculated in the respective body's coordinate system. This 
avoids the need to recompute them as bodies move around the simulation domain. The *normals* keyword is used to calculate these in the global coordinate system to be
exported through the :doc:`dump <dump>` command. The resulting per-atom x,y and z normals can be accessed through f_ID[9], f_ID[10], and f_ID[11], respectively.

As particles abrade and reduce in volume, the surface atoms storing the triangulation will converge closer together. For sufficient crowding, the areas of the triangulation will inflate the decomposed stresses, resulting in instability.  
The *remesh* keyword addresses this by removing surface spheres whose associated area falls below a threshold value. This value is set as the global minimum initial associated area multiplied by the *coeff* argument. For a *coeff* of 0, the associated
areas will never fall below the threshold. As the *coeff* approaches 1, re-meshing becomes increasingly frequent. Upon removing a surface atom, the affected triangulation is reconstructed to maintain a closed surface.

.. note::

   When using the *remesh* keyword, neighbor lists are rebuilt on any timestep during which a surface atom is removed. For 
   frequent re-meshing, this incurs a computational cost which may or may not be recovered by the now reduced number of 
   atoms in the simulation.  

The *equalise* keyword can be used in conjunction with the *remesh* keyword to re-mesh particles on timestep 0. In
this case, surface atoms with the smallest associated areas are recursively removed to minimise the standard deviation in areas 
across the particle's surface. This often acts to reduce the number of atoms in the simulation while allowing for a 
more consistent decomposition of surface stresses. The *equalise* keyword only takes place on timestep 0 of a simulation
and so will have no effect for subsequent :doc:`run <run>` commands or when starting from a restart file. 

The *stationary* keyword disables the integration of any particle's position and rotation. Surface atoms will still 
abrade if applied stresses exceed the respective hardnesses.

By default, the triangulation is referenced from the centers of the surface atoms corresponding to the positions 
outlined in the respective molecule file. As such, the surface seen by the granular pairwise models extends outwards
from the surface described by the triangulation. The *offset* keyword can be used to minimise this discrepancy. When 
used, surface atoms are offset towards their particle's center of mass by their radius. 

.. warning::

   When restarting simulations from a restart file, or using subsequent :doc:`run <run>` commands, the *offset* command must be kept consistent to ensure that rigid-body properties are correctly calculated. 

When applying gravitational forces to *rigid/abrade* bodies using the :doc:`fix gravity <fix_gravity>` command, the *gravity* keyword should also be used. Additionally, the :doc:`fix gravity <fix_gravity>` command
should not include any atoms in rigid bodies. In contrast to the *rigid/small* style, there is no restriction on using bodies with overlap between the constituent atoms.

When restarting a simulation from a restart file, the *infile* keyword should be used with the corresponding .rigid restart file. This allows the velocity, angular momentum, and 
total abraded volume of the rigid bodies to be conserved between simulations. The .rigid file also stores cumulative 
per-atom data such as the total displacement amount. If no .rigid file is provided through the *infile* keyword when using the :doc:`read_restart <read_restart>` command, 
the velocity and angular momentum of the bodies will be approximated from the velocity of their surface atoms. Other per-body information, such as the total abraded volume, is not recovered in this case.

.. note::

  Per-atom associated areas and cumulative displacement distances can be accsessed through f_ID[4], f_ID[8] respectively.

------------------------------------------------------------------------------------------

Restart, fix_modify, output, run start/stop, minimize info
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

When using the :doc:`restart <restart>`, an additional restart file with the suffix .rigid is produced. This file stores velocity and angular momentum of rigid bodies and other *rigid/abrade* specific properties. This file should be used with the optional *infile* keyword when restarting simulations to ensure continuity in cumulative values, e.g., total abraded volume, between runs. 

----------

Restrictions
""""""""""""

These fixes are all part of the RIGID package.  It is only enabled if
LAMMPS was built with that package. The *rigid/abrade* style additionally requires the MOLECULE package. See the :doc:`Build package
<Build_package>` page for more info.

The *rigid/abrade* style does not support adding rigid
bodies on-the-fly as molecules through fixes such as :doc:`fix deposit <fix_deposit>`
or :doc:`fix pour <fix_pour>`. The *mol* optional keyword from the *rigid/small* style has been removed. 

Related commands
""""""""""""""""
:doc:`fix_rigid <fix_rigid>`

Default
"""""""

none

----------

.. _Capozza:

**(Capozza)** R. Capozza, K. J. Hanley, A comprehensive model of plastic wear based on the discrete element method, Powder Technol, 410, 117864 (2022).

.. _Tonon:

**(Tonon)** F. Tonon, Explicit Exact Formulas for the 3-D Tetrahedron Inertia Tensor in Terms of its Vertex Coordinates, J. Math. Stat, 1 (2004).
