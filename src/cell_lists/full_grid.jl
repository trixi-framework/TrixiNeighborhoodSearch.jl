"""
    FullGridCellList(; min_corner, max_corner, search_radius = 0.0, periodicity = false)

A simple cell list implementation where each (empty or non-empty) cell of a rectangular
(axis-aligned) domain is assigned a list of points.
This cell list only works when all points are inside the specified domain at all times.

Only set `min_corner` and `max_corner` and use the default values for the other arguments
to create an empty "template" cell list that can be used to create an empty "template"
neighborhood search.
See [`copy_neighborhood_search`](@ref) for more details.

# Keywords
- `min_corner`: Coordinates of the domain corner in negative coordinate directions.
- `max_corner`: Coordinates of the domain corner in positive coordinate directions.
- `search_radius = 0.0`: Search radius of the neighborhood search, which will determine the
                         cell size. Use the default of `0.0` to create a template (see above).
- `periodicity = false`: Set to `true` when using a [`PeriodicBox`](@ref) with the
                         neighborhood search. When using [`copy_neighborhood_search`](@ref),
                         this option can be ignored an will be set automatically depending
                         on the periodicity of the neighborhood search.
"""
struct FullGridCellList{C, LI, MC} <: AbstractCellList
    cells          :: C
    linear_indices :: LI
    min_cell       :: MC

    function FullGridCellList{C, LI, MC}(cells, linear_indices, min_cell) where {C, LI, MC}
        new{C, LI, MC}(cells, linear_indices, min_cell)
    end
end

function FullGridCellList(; min_corner, max_corner, search_radius = 0.0,
                          periodicity = false,
                          backend = DynamicVectorOfVectors{Int32}(max_outer_length = 0,
                                                                  max_inner_length = 100))
    if search_radius < eps()
        # Create an empty "template" cell list to be used with `copy_cell_list`
        cells = nothing
        linear_indices = nothing

        # Misuse `min_cell` to store min and max corner for copying
        min_cell = (min_corner, max_corner)
    else
        if periodicity
            # Subtract `min_corner` because that's how the grid NHS works with periodicity
            max_corner = max_corner .- min_corner
            min_corner = min_corner .- min_corner
        end

        # Note that we don't shift everything so that the first cell starts at `min_corner`.
        # The first cell is the cell containing `min_corner`, so we need to add one layer
        # in order for `max_corner` to be inside a cell.
        n_cells_per_dimension = ceil.(Int, (max_corner .- min_corner) ./ search_radius) .+ 1
        linear_indices = LinearIndices(Tuple(n_cells_per_dimension))
        min_cell = Tuple(floor_to_int.(min_corner ./ search_radius))

        cells = construct_backend(backend, n_cells_per_dimension)
    end

    return FullGridCellList{typeof(cells), typeof(linear_indices),
                            typeof(min_cell)}(cells, linear_indices, min_cell)
end

function construct_backend(::Vector{Vector{T}}, size) where {T}
    return [T[] for _ in 1:prod(size)]
end

function construct_backend(cells::DynamicVectorOfVectors{T}, size_) where {T}
    # Create new `DynamicVectorOfVectors` with the same `max_inner_length`
    max_inner_length = size(cells.backend, 1)
    cells = DynamicVectorOfVectors{T}(max_outer_length = prod(size_); max_inner_length)
    resize!(cells, prod(size_))

    return cells
end

function Base.empty!(cell_list::FullGridCellList)
    (; cells) = cell_list

    # `Base.empty!.(cells)`, but for all backends
    for i in eachindex(cells)
        emptyat!(cells, i)
    end

    return cell_list
end

function Base.empty!(cell_list::FullGridCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    throw(UndefRefError("`search_radius` is not defined for this cell list"))
end

function push_cell!(cell_list::FullGridCellList, cell, particle)
    (; cells) = cell_list

    # `push!(cell_list[cell], particle)`, but for all backends
    pushat!(cells, cell_index(cell_list, cell), particle)

    return cell_list
end

function push_cell!(cell_list::FullGridCellList{Nothing}, cell, particle)
    # This is an empty "template" cell list to be used with `copy_cell_list`
    throw(UndefRefError("`search_radius` is not defined for this cell list"))
end

function deleteat_cell!(cell_list::FullGridCellList, cell, i)
    (; cells) = cell_list

    # `deleteat!(cell_list[cell], i)`, but for all backends
    deleteatat!(cells, cell_index(cell_list, cell), i)
end

@inline each_cell_index(cell_list::FullGridCellList) = eachindex(cell_list.cells)

function each_cell_index(cell_list::FullGridCellList{Nothing})
    # This is an empty "template" cell list to be used with `copy_cell_list`
    throw(UndefRefError("`search_radius` is not defined for this cell list"))
end

@inline function cell_index(cell_list::FullGridCellList, cell::Tuple)
    (; linear_indices, min_cell) = cell_list

    return linear_indices[(cell .- min_cell .+ 1)...]
end

@inline cell_index(::FullGridCellList, cell::Integer) = cell

@inline function Base.getindex(cell_list::FullGridCellList, cell)
    (; cells) = cell_list

    return cells[cell_index(cell_list, cell)]
end

@inline function is_correct_cell(cell_list::FullGridCellList, cell_coords, cell_index_)
    return cell_index(cell_list, cell_coords) == cell_index_
end

@inline index_type(::FullGridCellList) = Int32

function copy_cell_list(cell_list::FullGridCellList, search_radius, periodic_box)
    # Misuse `min_cell` to store min and max corner for copying
    min_corner, max_corner = cell_list.min_cell

    return FullGridCellList(; min_corner, max_corner, search_radius,
                            periodicity = !isnothing(periodic_box))
end
