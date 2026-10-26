#!/usr/bin/env Rscript
# Convert the WGCNA module memberships from .RData to CSV so Python can read them.
#
# The published file is an R workspace holding two data.tables:
#   cur.map  gene -> module membership (3,879 rows: cur_labels, display_label, kme, prev_modules)
#   mod.map  module -> colour crosswalk (14 rows, old WGCNA colour names to M0..M23 labels)
#
# M0 is the "grey" catch-all of 2,091 unassigned genes -- 13 modules are real.
#
# Usage:  Rscript scripts/export_wgcna.R

src  <- "data/raw/wgcna/merged_older_wgcna_kme.RData"
dest <- "data/interim"

if (!file.exists(src)) stop("missing ", src, " -- run `bash scripts/get_data.sh` first")
dir.create(dest, showWarnings = FALSE, recursive = TRUE)

e <- new.env()
load(src, envir = e)

stopifnot(all(c("cur.map", "mod.map") %in% ls(e)))

cur <- as.data.frame(get("cur.map", envir = e))
mod <- as.data.frame(get("mod.map", envir = e))

write.csv(cur, file.path(dest, "wgcna_cur_map.csv"), row.names = FALSE)
write.csv(mod, file.path(dest, "wgcna_mod_map.csv"), row.names = FALSE)

cat("wrote", file.path(dest, "wgcna_cur_map.csv"), "-", nrow(cur), "genes\n")
cat("wrote", file.path(dest, "wgcna_mod_map.csv"), "-", nrow(mod), "modules\n\n")
cat("genes per module:\n")
print(table(cur$cur_labels))
