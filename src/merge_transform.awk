#!/usr/bin/awk -f


# Print the opening line for the MLIR module with the required attribute
BEGIN { print "module attributes {transform.with_named_sequence} {" }

# If this is the first line of a file (but not the first file overall), print a blank line to separate modules
FNR==1 && NR!=1 { print "" }

# Remove the outer 'module {' and its matching closing '}' from each file
/module[[:space:]]*{/ {
    in_module = 1
    next
}
in_module && /^\}/ {
    in_module = 0
    next
}
in_module {
    print
}

# After all files have been processed, print the closing brace for the module
END { print "}" }
