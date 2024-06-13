# hip-arrowhead-kernel
GPU kernel for directly solving an arrowhead matrix written with HIP

# Overview
In this code, an arrowhead matrix is a sparse matrix which contains nonzeros on the diagonal, the bottom row, and the right-most column.  This sparsity pattern, and variations of it, arise when solving different variations of the Boltzmann transport equation (BTE) using the Coupled Ordinates Method (COMET).  Variations of the BTE include those for modeling phonon transport, molecular gas dynamics, electron transport, and radiative heat transfer.  Each variation leads to a slight difference to the matrix (i.e., the bottom two rows and the 2 right most columns are nonzero).

The solution method for the BTE in this context is the finite volume method which gives us a linear system for each cell that yields an arrowhead matrix which can be solved directly.  So, for a finite volume mesh containing `nCells` cells, there will be `nCells` arrowhead matrices.  Each arrowhead matrix will contain `nKcells + 1` rows, where `nKcells` is the discretization in the 3-dimensional phase space.  This example will show how to solve each arrowhead linear system for each cell simultaneously for all `nCells` cells of a mesh.

# Solution
Given an arrowhead matrix linear system:

$$\begin{matrix}
d_1 x_1 + &     &      &  c_1 x_l  \\
    & d_2 x_2 + &        & c_2 x_l  \\
    &     & \ddots & \vdots\\
r_1 x_1 + & r_2 x_2 +& \dots  & c_l x_l
\end{matrix}
\begin{matrix}
=b_1\\
=b_2\\
\vdots\\
=b_l
\end{matrix}$$

we can solve for each $x_i$ ($i\neq l$):

$$x_i = -\frac{c_i}{d_i}x_l + \frac{b_i}{d_i}$$

which we can then substitue into the bottom equation, combine terms, and rearrange into an equation for $x_l$:

$$x_l = \frac{
  b_l - \sum\limits_{i=1}^{l-1}\frac{r_i b_i}{d_i}
}{
  c_l - \sum\limits_{i=1}^{l-1}\frac{r_i c_i}{d_i}
}$$

Thus one solution procedure is to solve for $x_l$ and then solve for the remaining $x_i$ values.

# Code Descripition
The code is split out into 3 files, the entrypoint `arrowhead.cpp`, the kernel definitions `kernels.cpp`, and utilities `utils.hpp`.

## Problem Setup
The problem we're solving will have `nCells` cells, each containing an arrowhead linear system containing `nKcells + 1` equations.  The majority of the initialization takes place inside the `ArrowheadSystem` class inside `utils.hpp`.  Each matrix is randomly initialized such that it will be non-singular.  The solution is randomly initialized as well, and is in turn used to compute the corresponding source term (the $b_i$ values).

## Algorithm
### Solving $x_l$
Solving for $x_l$ is mainly comprised of 2 reduction operations across `nKcells` values, followed by a single division operation.  The reduction for each cells takes place on its own block to take advantage of fast shared memory.  The reductions are performed in a tree like manner making use of sequential addressing to reduce memory bank conflicts.
### Solving for $x_i$
Solving for $x_i$ is much more straighforward as it is embarassingly parallel.  Each computation is distributed across blocks.  The number of blocks is determined based on the number of computations and a preconfigured number of threads per block.