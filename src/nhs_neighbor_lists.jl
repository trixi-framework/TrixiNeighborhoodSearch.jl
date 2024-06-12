@doc raw"""
    NeighborListsNeighborhoodSearch{NDIMS}(search_radius, n_particles;
                                           periodic_box_min_corner = nothing,
                                           periodic_box_max_corner = nothing,
                                           backend = VectorOfVectors{Int32})

Neighborhood search with precomputed neighbor lists. A list of all neighbors is computed
for each particle during initialization and update.
This neighborhood search maximizes the performance of neighbor loops at the cost of a much
slower [`update!`](@ref).

A [`GridNeighborhoodSearch`](@ref) is used internally to compute the neighbor lists during
initialization and update.

# Arguments
- `NDIMS`:          Number of dimensions.
- `search_radius`:  The uniform search radius.
- `n_particles`:    Total number of particles.

# Keywords
- `periodic_box_min_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in negative coordinate
                                directions.
- `periodic_box_max_corner`:    In order to use a (rectangular) periodic domain, pass the
                                coordinates of the domain corner in positive coordinate
                                directions.
- `backend=VectorOfVectors{Int32}`: Data structure to store the neighbor lists. The default
                                    `VectorOfVectors` is a data structure from
                                    [ArraysOfArrays.jl](https://github.com/JuliaArrays/ArraysOfArrays.jl),
                                    which behaves like a `Vector` of `Vector`s, but uses
                                    a single `Vector` for a contiguous memory layout.
                                    Use `backend=Vector{Vector{Int32}}` to benchmark
                                    the benefits of this representation.
"""
struct NeighborListsNeighborhoodSearch{NDIMS, NHS, NL, PB}
    neighborhood_search :: NHS
    neighbor_lists      :: NL
    periodic_box        :: PB

    function NeighborListsNeighborhoodSearch{NDIMS}(search_radius, n_particles;
                                                    periodic_box_min_corner = nothing,
                                                    periodic_box_max_corner = nothing,
                                                    backend = VectorOfVectors{Int32}) where {
                                                                                             NDIMS
                                                                                             }
        nhs = GridNeighborhoodSearch{NDIMS}(search_radius, n_particles,
                                            periodic_box_min_corner = periodic_box_min_corner,
                                            periodic_box_max_corner = periodic_box_max_corner)

        neighbor_lists = backend()

        new{NDIMS, typeof(nhs), typeof(neighbor_lists),
            typeof(nhs.periodic_box)}(nhs, neighbor_lists, nhs.periodic_box)
    end
end

@inline function Base.ndims(::NeighborListsNeighborhoodSearch{NDIMS}) where {NDIMS}
    return NDIMS
end

function initialize!(search::NeighborListsNeighborhoodSearch,
                     x::AbstractMatrix, y::AbstractMatrix)
    (; neighborhood_search, neighbor_lists) = search

    # Initialize grid NHS
    initialize!(neighborhood_search, x, y)

    initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
end

function update!(search::NeighborListsNeighborhoodSearch,
                 x::AbstractMatrix, y::AbstractMatrix;
                 particles_moving = (true, true))
    (; neighborhood_search, neighbor_lists) = search

    # Update grid NHS
    update!(neighborhood_search, x, y, particles_moving = particles_moving)

    # Skip update if both point sets are static
    if any(particles_moving)
        initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
    end
end

function initialize_neighbor_lists!(neighbor_lists::Vector{<:Vector}, neighborhood_search,
                                    x, y)
    # Initialize neighbor lists
    empty!(neighbor_lists)
    resize!(neighbor_lists, size(x, 2))
    for i in eachindex(neighbor_lists)
        neighbor_lists[i] = Int[]
    end

    # Fill neighbor lists
    for_particle_neighbor(x, y, neighborhood_search) do particle, neighbor, _, _
        push!(neighbor_lists[particle], neighbor)
    end
end

function initialize_neighbor_lists!(neighbor_lists, neighborhood_search, x, y)
    neighbor_lists_ = Vector{Vector{Int32}}()
    initialize_neighbor_lists!(neighbor_lists_, neighborhood_search, x, y)

    empty!(neighbor_lists)
    for i in eachindex(neighbor_lists_)
        push!(neighbor_lists, neighbor_lists_[i])
    end
end

@inline function foreach_neighbor(f, system_coords, neighbor_system_coords,
                                  neighborhood_search::NeighborListsNeighborhoodSearch,
                                  particle; search_radius = nothing)
    (; periodic_box, neighbor_lists) = neighborhood_search
    (; search_radius) = neighborhood_search.neighborhood_search

    particle_coords = extract_svector(system_coords, Val(ndims(neighborhood_search)),
                                      particle)
    for neighbor in neighbor_lists[particle]
        neighbor_coords = extract_svector(neighbor_system_coords,
                                          Val(ndims(neighborhood_search)), neighbor)

        pos_diff = particle_coords - neighbor_coords
        distance2 = dot(pos_diff, pos_diff)

        pos_diff, distance2 = compute_periodic_distance(pos_diff, distance2, search_radius,
                                                        periodic_box)

        distance = sqrt(distance2)

        # Inline to avoid loss of performance
        # compared to not using `for_particle_neighbor`.
        @inline f(particle, neighbor, pos_diff, distance)
    end
end
