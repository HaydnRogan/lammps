.. index:: compute rigid/local_abrade

compute rigid/local_abrade command
==================================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID rigid/local_abrade rigidID input1 input2 ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* rigid/local_abrade = style name of this compute command
* rigidID = ID of fix rigid/abrade command
* input = one or more rigid body attributes

  .. parsed-literal::

       possible attributes = id, mol, mass, volume,
                             surface_area, abraded_volume, wear_energy,
                             x, y, z, xu, yu, zu, ix, iy, iz,
                             vx, vy, vz, fx, fy, fz,
                             omegax, omegay, omegaz,
                             angmomx, angmomy, angmomz,
                             quatw, quati, quatj, quatk,
                             tqx, tqy, tqz,
                             inertiax, inertiay, inertiaz

  .. parsed-literal::

           id = atom ID of atom within body which owns body properties
           mol = molecule ID used to define body in :doc:`fix rigid/small <fix_rigid>` command
           volume = total volume of body
           mass = total mass of body
           surface_area = total surface area of the body
           abraded_volume = cumulative volume removed from the particle 
           wear_energy = cumulative dissipated energy
           x,y,z = center of mass coords of body
           xu,yu,zu = unwrapped center of mass coords of body
           ix,iy,iz = box image that the center of mass is in
           vx,vy,vz = center of mass velocities
           fx,fy,fz = force of center of mass
           omegax,omegay,omegaz = angular velocity of body
           angmomx,angmomy,angmomz = angular momentum of body
           quatw,quati,quatj,quatk = quaternion components for body
           tqx,tqy,tqz = torque on body
           inertiax,inertiay,inertiaz = diagonalized moments of inertia of body

Examples
""""""""

.. code-block:: LAMMPS

   compute 1 all rigid/local_abrade abrade id volume surface_area abraded_volume wear_energy

Description
"""""""""""


An extension of the :doc:`rigid/local <compute_rigid_local>` command, *compute rigid/local_abrade* grants 
access to abrasion-specific rigid body attributes for rigid bodies defined by the :doc:`rigid/abrade <fix_rigid_abrade>`
command. The data is stored as local data so it can be accessed by other :doc:`output commands <Howto_output>` that 
process local data, such as the :doc:`compute reduce <compute_reduce>` or :doc:`dump local <dump>` commands.

Here is an example of how to use this compute to dump rigid body info to a file:

.. code-block:: LAMMPS

   compute 1 all rigid/local_abrade abrade id mass volume surface_area abraded_volume wear_energy
   dump 2 all local 500 dump.abrade index c_3[1] c_3[2] c_3[3] c_3[4] c_3[5] c_3[6] 

----------

This section explains the abrasion-specific rigid body attributes that can be specified.

The *volume* attribute is the total volume of the body as calculated through the triangulation stored in the :doc:`angles <angles>` comprising the surface atoms.

The *mass* is the product of the body's volume and mass density. 

The *surface_area* is the sum of associated areas for all atoms in the body.

The *abraded_volume* is the total abraded volume removed from the body. This cumulative value can be preserved between simulations through the use of the optional *infile* keyword in the :doc:`rigid/abrade <fix_rigid_abrade>` command.

The *wear_energy* is the total energy dissipated to displace surface atoms inwards along their normals. The energy dissipated to displace a surface atom :math:`i` is adapted from :ref:`(Capozza) <Capozza_compute>` as,

.. math::

   E_{w,i} = \left| \mathbf{F}_{i,n} \right| \left| \mathbf{u}_i \right| \Delta t

where, :math:`\mathbf{F}_{i,n}` is the imposed force aligned along the atom's associate normal, :math:`\mathbf{u}_i` is the total abrasion velocity, and :math:`\Delta t` is the employed timestep.

----------

Output info
"""""""""""

This compute calculates a local vector or local array depending on the
number of keywords.  The length of the vector or number of rows in the
array is the number of rigid bodies.  If a single keyword is
specified, a local vector is produced.  If two or more keywords are
specified, a local array is produced where the number of columns = the
number of keywords.  The vector or array can be accessed by any
command that uses local values from a compute as input.  See the
:doc:`Howto output <Howto_output>` page for an overview of LAMMPS
output options.

The vector or array values will be in whatever :doc:`units <units>` the
corresponding attribute is in:

* id,mol = unitless
* mass = mass units
* volume = volume units
* surface_area = area units
* abraded_volume = volume units
* wear_energy = energy units
* x,y,z and xy,yu,zu = distance units
* vx,vy,vz = velocity units
* fx,fy,fz = force units
* omegax,omegay,omegaz = radians/time units
* angmomx,angmomy,angmomz = mass\*distance\ :math:`^2`\ /time units
* quatw,quati,quatj,quatk = unitless
* tqx,tqy,tqz = torque units
* inertiax,inertiay,inertiaz = mass\*distance\ :math:`^2` units

Restrictions
""""""""""""

This compute is part of the RIGID package.  It is only enabled if
LAMMPS was built with that package.  See the :doc:`Build package <Build_package>` page for more info.

Related commands
""""""""""""""""

:doc:`rigid/local <compute_rigid_local>`, :doc:`dump local <dump>`, :doc:`compute reduce <compute_reduce>`

Default
"""""""

none

----------

.. _Capozza_compute:

**(Capozza)** R. Capozza, K. J. Hanley, A comprehensive model of plastic wear based on the discrete element method, Powder Technol, 410, 117864 (2022).
