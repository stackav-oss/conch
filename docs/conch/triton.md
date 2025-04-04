# Triton Best Practices

1. Constexpr parameters (`tl.constexpr`) should be `snake_case`, with a leading `cxpr_` prefix.
    - This convention is not widely recognized, as much existing Triton literature uses `SCREAMING_SNAKE_CASE` for `tl.constexpr`.
    However, we believe that this notation is more user-fiendly, as it clearly designates these parameters as constexpr and they cannot be easily mistaken for other Python constants.
1. Global constants (those marked `tl.constexpr` outside of a kernel) should be `SCREAMING_SNAKE_CASE` with a leading underscore.
    - These constants are much more similar to Python constants, and they should be named as such.
    This can be useful for things like using Enum values as constants.
1. Tensor parameters (`tl.tensor`) should be `snake_case`, with a trailing `_ptr` suffix.
    - This convention is also not widely recognized, and there is no standard by prevailing Triton literature.
    - Similar to the `constexpr` naming convention, this dismbiguates which parameters are pointers that must be `tl.load()`-ed.
1. Stride parameters should be `snake_case` and contain `stride` in the parameter name.
1. Scalar parameters should be `snake_case`.
