var documenterSearchIndex = {"docs":
[{"location":"license/","page":"License","title":"License","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/LICENSE.md\"","category":"page"},{"location":"license/#License","page":"License","title":"License","text":"","category":"section"},{"location":"license/","page":"License","title":"License","text":"MIT LicenseCopyright (c) 2023-present The TrixiParticles.jl Authors (see Authors) \nCopyright (c) 2023-present Helmholtz-Zentrum hereon GmbH, Institute of Surface Science \n \nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.","category":"page"},{"location":"authors/","page":"Authors","title":"Authors","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/AUTHORS.md\"","category":"page"},{"location":"authors/#Authors","page":"Authors","title":"Authors","text":"","category":"section"},{"location":"authors/","page":"Authors","title":"Authors","text":"This package is maintained by the authors of TrixiParticles.jl. For a full list of authors, see AUTHORS.md in the TrixiParticles.jl repository. These authors form \"The TrixiParticles.jl Authors\", as mentioned under License.","category":"page"},{"location":"reference/#API-reference","page":"API reference","title":"API reference","text":"","category":"section"},{"location":"reference/","page":"API reference","title":"API reference","text":"CurrentModule = PointNeighbors","category":"page"},{"location":"reference/","page":"API reference","title":"API reference","text":"Modules = [PointNeighbors]","category":"page"},{"location":"reference/#PointNeighbors.GridNeighborhoodSearch","page":"API reference","title":"PointNeighbors.GridNeighborhoodSearch","text":"GridNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,\n                              periodic_box = nothing, threaded_update = true)\n\nSimple grid-based neighborhood search with uniform search radius. The domain is divided into a regular grid. For each (non-empty) grid cell, a list of points in this cell is stored. Instead of representing a finite domain by an array of cells, a potentially infinite domain is represented by storing cell lists in a hash table (using Julia's Dict data structure), indexed by the cell index tuple\n\nleft( leftlfloor fracxd rightrfloor leftlfloor fracyd rightrfloor right) quad textor quad\nleft( leftlfloor fracxd rightrfloor leftlfloor fracyd rightrfloor leftlfloor fraczd rightrfloor right)\n\nwhere x y z are the space coordinates and d is the search radius.\n\nTo find points within the search radius around a position, only points in the neighboring cells are considered.\n\nSee also (Chalela et al., 2021), (Ihmsen et al. 2011, Section 4.4).\n\nAs opposed to (Ihmsen et al. 2011), we do not sort the points in any way, since not sorting makes our implementation a lot faster (although less parallelizable).\n\nArguments\n\nNDIMS: Number of dimensions.\n\nKeywords\n\nsearch_radius = 0.0:    The fixed search radius. The default of 0.0 is useful together                           with copy_neighborhood_search.\nn_points = 0:           Total number of points. The default of 0 is useful together                           with copy_neighborhood_search.\nperiodic_box = nothing: In order to use a (rectangular) periodic domain, pass a                           PeriodicBox.\nthreaded_update = true: Can be used to deactivate thread parallelization in the                           neighborhood search update. This can be one of the largest                           sources of variations between simulations with different                           thread numbers due to neighbor ordering changes.\n\nReferences\n\nM. Chalela, E. Sillero, L. Pereyra, M.A. Garcia, J.B. Cabral, M. Lares, M. Merchán. \"GriSPy: A Python package for fixed-radius nearest neighbors search\". In: Astronomy and Computing 34 (2021). doi: 10.1016/j.ascom.2020.100443\nMarkus Ihmsen, Nadir Akinci, Markus Becker, Matthias Teschner. \"A Parallel SPH Implementation on Multi-Core CPUs\". In: Computer Graphics Forum 30.1 (2011), pages 99–112. doi: 10.1111/J.1467-8659.2010.01832.X\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.PeriodicBox","page":"API reference","title":"PointNeighbors.PeriodicBox","text":"PeriodicBox(; min_corner, max_corner)\n\nDefine a rectangular periodic domain.\n\nKeywords\n\nmin_corner: Coordinates of the domain corner in negative coordinate directions.\nmax_corner: Coordinates of the domain corner in positive coordinate directions.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.PrecomputedNeighborhoodSearch","page":"API reference","title":"PointNeighbors.PrecomputedNeighborhoodSearch","text":"PrecomputedNeighborhoodSearch{NDIMS}(; search_radius = 0.0, n_points = 0,\n                                     periodic_box = nothing, threaded_update = true)\n\nNeighborhood search with precomputed neighbor lists. A list of all neighbors is computed for each point during initialization and update. This neighborhood search maximizes the performance of neighbor loops at the cost of a much slower update!.\n\nA GridNeighborhoodSearch is used internally to compute the neighbor lists during initialization and update.\n\nArguments\n\nNDIMS: Number of dimensions.\n\nKeywords\n\nsearch_radius = 0.0:    The fixed search radius. The default of 0.0 is useful together                           with copy_neighborhood_search.\nn_points = 0:           Total number of points. The default of 0 is useful together                           with copy_neighborhood_search.\nperiodic_box = nothing: In order to use a (rectangular) periodic domain, pass a                           PeriodicBox.\nthreaded_update = true: Can be used to deactivate thread parallelization in the                           neighborhood search update. This can be one of the largest                           sources of variations between simulations with different                           thread numbers due to neighbor ordering changes.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.TrivialNeighborhoodSearch","page":"API reference","title":"PointNeighbors.TrivialNeighborhoodSearch","text":"TrivialNeighborhoodSearch{NDIMS}(; search_radius = 0.0, eachpoint = 1:0,\n                                 periodic_box = nothing)\n\nTrivial neighborhood search that simply loops over all points.\n\nArguments\n\nNDIMS: Number of dimensions.\n\nKeywords\n\nsearch_radius = 0.0:    The fixed search radius. The default of 0.0 is useful together                           with copy_neighborhood_search.\neachpoint = 1:0:        Iterator for all point indices. Usually just 1:n_points.                           The default of 1:0 is useful together with                           copy_neighborhood_search.\nperiodic_box = nothing: In order to use a (rectangular) periodic domain, pass a                           PeriodicBox.\n\n\n\n\n\n","category":"type"},{"location":"reference/#PointNeighbors.copy_neighborhood_search-Tuple{PointNeighbors.AbstractNeighborhoodSearch, Any, Any}","page":"API reference","title":"PointNeighbors.copy_neighborhood_search","text":"copy_neighborhood_search(search::AbstractNeighborhoodSearch, search_radius, n_points;\n                         eachpoint = 1:n_points)\n\nCreate a new uninitialized neighborhood search of the same type and with the same configuration options as search, but with a different search radius and number of points.\n\nThe TrivialNeighborhoodSearch also requires an iterator eachpoint, which most of the time will be 1:n_points. If the TrivialNeighborhoodSearch is never going to be used, the keyword argument eachpoint can be ignored.\n\nThis is useful when a simulation code requires multiple neighborhood searches of the same kind. One can then just pass an empty neighborhood search as a template and use this function inside the simulation code to generate similar neighborhood searches with different search radii and different numbers of points.\n\n# Template\nnhs = GridNeighborhoodSearch{2}()\n\n# Inside the simulation code, generate similar neighborhood searches\nnhs1 = copy_neighborhood_search(nhs, 1.0, 100)\n\n# output\nGridNeighborhoodSearch{2, Float64, ...}(...)\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.initialize!-Tuple{PointNeighbors.AbstractNeighborhoodSearch, Any, Any}","page":"API reference","title":"PointNeighbors.initialize!","text":"initialize!(search::AbstractNeighborhoodSearch, x, y)\n\nInitialize a neighborhood search with the two coordinate arrays x and y.\n\nIn general, the purpose of a neighborhood search is to find for one point in x all points in y whose distances to that point are smaller than the search radius. x and y are expected to be matrices, where the i-th column contains the coordinates of point i. Note that x and y can be identical.\n\nSee also update!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.update!-Tuple{PointNeighbors.AbstractNeighborhoodSearch, Any, Any}","page":"API reference","title":"PointNeighbors.update!","text":"update!(search::AbstractNeighborhoodSearch, x, y; points_moving = (true, true))\n\nUpdate an already initialized neighborhood search with the two coordinate arrays x and y.\n\nLike initialize!, but reusing the existing data structures of the already initialized neighborhood search. When the points only moved a small distance since the last update! or initialize!, this is significantly faster than initialize!.\n\nNot all implementations support incremental updates. If incremental updates are not possible for an implementation, update! will fall back to a regular initialize!.\n\nSome neighborhood searches might not need to update when only x changed since the last update! or initialize! and y did not change. Pass points_moving = (true, false) in this case to avoid unnecessary updates. The first flag in points_moving indicates if points in x are moving. The second flag indicates if points in y are moving.\n\nSee also initialize!.\n\n\n\n\n\n","category":"method"},{"location":"reference/#PointNeighbors.@threaded-Tuple{Any}","page":"API reference","title":"PointNeighbors.@threaded","text":"@threaded for ... end\n\nSemantically the same as Threads.@threads when iterating over a AbstractUnitRange but without guarantee that the underlying implementation uses Threads.@threads or works for more general for loops. In particular, there may be an additional check whether only one thread is used to reduce the overhead of serial execution or the underlying threading capabilities might be provided by other packages such as Polyester.jl.\n\nwarn: Warn\nThis macro does not necessarily work for general for loops. For example, it does not necessarily support general iterables such as eachline(filename).\n\nSome discussion can be found at https://discourse.julialang.org/t/overhead-of-threads-threads/53964 and https://discourse.julialang.org/t/threads-threads-with-one-thread-how-to-remove-the-overhead/58435.\n\nCopied from Trixi.jl.\n\n\n\n\n\n","category":"macro"},{"location":"","page":"Home","title":"Home","text":"EditURL = \"https://github.com/trixi-framework/PointNeighbors.jl/blob/main/README.md\"","category":"page"},{"location":"#PointNeighbors.jl","page":"Home","title":"PointNeighbors.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Docs-stable) (Image: Docs-dev) (Image: Slack) (Image: Youtube) (Image: Build Status) (Image: Codecov) (Image: SciML Code Style) (Image: License: MIT)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Work in Progress!","category":"page"}]
}
