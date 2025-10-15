# Claude Code Style Guide for COIN_equality

## Coding Philosophy
This project prioritizes elegant, fail-fast code that surfaces errors quickly rather than hiding them.

## Core Style Requirements

### Error Handling
- No input validation on function parameters (except for command-line interfaces)
- No defensive programming - let exceptions bubble up naturally
- Fail fast - prefer code that crashes immediately on invalid inputs rather than continuing with bad data
- No try-catch blocks unless absolutely necessary for program logic (not error suppression)
- No optional function arguments - all parameters must be explicitly provided
- Assume complete data - do not check for missing data fields. If required data is missing, let the code fail with natural Python errors

### Code Elegance
- Minimize conditional statements - prefer functional approaches, mathematical expressions, and numpy vectorization
- Favor mathematical clarity over defensive checks
- Use numpy operations instead of loops and conditionals where possible
- Compute once, use many times - move invariant calculations outside loops and create centralized helper functions
- No backward compatibility - do not add conditional logic to support deprecated field names or old configurations. Update all code and configurations to use current conventions.
- Use standard Python packages - prefer established numerical methods from scipy, numpy, etc. rather than implementing custom numerical algorithms

### Code Organization
- All imports at the top of the file - no imports inside functions or scattered throughout the code


### Naming Conventions
- Consistent naming - use the same variable/field names throughout the codebase when referring to the same concept
- Descriptive names preferred - long, clear names are better than short, ambiguous ones
- Follow source naming - when accessing data from a structure (e.g., `config['response_function_scalings']['scaling_name']`), prefer using the same field name (`scaling_name`) or a composite (`response_function_scaling_name`) rather than inventing new descriptors
- Example: If config has `scaling_name` under `response_function_scalings`, use `scaling_name` or `response_function_scaling_name`, not `response_func` or other variations

### Function Design
- Functions should assume valid inputs and focus on their core mathematical/logical purpose
- Let Python's natural error messages guide debugging rather than custom error handling
- All function arguments must be explicitly provided - no default values (`=None`) or conditional logic
- Clean fail-fast approach - if required arguments are not supplied, the code should fail immediately with a clear error
