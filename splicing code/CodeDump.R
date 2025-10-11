library(data.table)

# strand can be 1/2 or '+'/'-'. We normalize to '+'/'-'.
.normalize_strand <- function(x) {
  if (is.numeric(x)) x <- as.integer(x)
  if (is.integer(x)) {
    out <- ifelse(x == 1L, "+", ifelse(x == 2L, "-", NA_character_))
  } else {
    out <- ifelse(x %in% c("+","1","plus"), "+",
           ifelse(x %in% c("-","2","minus"), "-", NA_character_))
  }
  out
}

# Use the external merged_feature table (your printout) + marvel$sj.metadata (optional)
.build_sj_index_from_feature <- function(MarvelObject, feature_dt) {
  feat <- as.data.table(feature_dt)
  stopifnot(all(c("chrom","intron_start","intron_end","strand","coord.intron") %in% names(feat)))

  feat[, strand := .normalize_strand(strand)]
  feat[, `:=`(
    start = as.integer(intron_start),
    end   = as.integer(intron_end)
  )]

  # donor(5'SS) / acceptor(3'SS) are defined by STRAND (not genomic left/right)
  feat[, donor    := ifelse(strand == "+", start, end)]
  feat[, acceptor := ifelse(strand == "+", end,   start)]

  # Also keep genomic left/right for convenience in event detection
  feat[, `:=`(left = pmin(start, end), right = pmax(start, end))]

  # Attach per-SJ gene labels if available in MARVEL
  if (!is.null(MarvelObject$sj.metadata)) {
    sjmd <- as.data.table(MarvelObject$sj.metadata)
    # Expected columns include coord.intron, gene_short_name.start/end, sj.type
    feat <- merge(feat, sjmd, by = "coord.intron", all.x = TRUE)
  }

  # Keep only SJs that exist in the matrix
  present <- rownames(MarvelObject$sj.count.matrix)
  feat <- feat[coord.intron %in% present]

  setkey(feat, chrom, strand, donor, acceptor)
  feat[]
}

psi_donor_usage_strand <- function(MarvelObject, feature_dt,
                                   gene_of_interest = NULL,
                                   min_cell_total = 0L) {
  idx <- .build_sj_index_from_feature(MarvelObject, feature_dt)

  if (!is.null(gene_of_interest)) {
    has_gene_cols <- c("gene_short_name.start","gene_short_name.end") %in% names(idx)
    if (all(has_gene_cols)) {
      idx <- idx[(gene_short_name.start == gene_of_interest) | (gene_short_name.end == gene_of_interest)]
    }
    if (nrow(idx) == 0L) stop("No SJs found for gene: ", gene_of_interest)
  }

  mat <- MarvelObject$sj.count.matrix
  stopifnot(identical(colnames(mat), colnames(MarvelObject$gene.count.matrix)))

  X  <- as.matrix(mat[idx$coord.intron, , drop = FALSE])
  dt <- as.data.table(as.table(X))
  setnames(dt, c("coord.intron","cell.id","sj.count"))
  dt <- merge(dt, idx[, .(coord.intron, chrom, strand, donor, acceptor)], by = "coord.intron")

  # denominator per (chrom, strand, donor, cell)
  dt[, denom := sum(sj.count), by = .(chrom, strand, donor, cell.id)]
  if (min_cell_total > 0L) dt[denom < min_cell_total, denom := 0L]
  dt[, PSI := fifelse(denom > 0, (sj.count / denom) * 100, NA_real_)]

  dt[, denom := NULL]
  setorder(dt, chrom, strand, donor, acceptor, cell.id)
  dt[]
}

psi_acceptor_usage_strand <- function(MarvelObject, feature_dt,
                                      gene_of_interest = NULL,
                                      min_cell_total = 0L) {
  idx <- .build_sj_index_from_feature(MarvelObject, feature_dt)

  if (!is.null(gene_of_interest)) {
    has_gene_cols <- c("gene_short_name.start","gene_short_name.end") %in% names(idx)
    if (all(has_gene_cols)) {
      idx <- idx[(gene_short_name.start == gene_of_interest) | (gene_short_name.end == gene_of_interest)]
    }
    if (nrow(idx) == 0L) stop("No SJs found for gene: ", gene_of_interest)
  }

  mat <- MarvelObject$sj.count.matrix
  stopifnot(identical(colnames(mat), colnames(MarvelObject$gene.count.matrix)))

  X  <- as.matrix(mat[idx$coord.intron, , drop = FALSE])
  dt <- as.data.table(as.table(X))
  setnames(dt, c("coord.intron","cell.id","sj.count"))
  dt <- merge(dt, idx[, .(coord.intron, chrom, strand, donor, acceptor)], by = "coord.intron")

  # denominator per (chrom, strand, acceptor, cell)
  dt[, denom := sum(sj.count), by = .(chrom, strand, acceptor, cell.id)]
  if (min_cell_total > 0L) dt[denom < min_cell_total, denom := 0L]
  dt[, PSI := fifelse(denom > 0, (sj.count / denom) * 100, NA_real_)]

  dt[, denom := NULL]
  setorder(dt, chrom, strand, acceptor, donor, cell.id)
  dt[]
}

.find_cassette_triplets_strand <- function(MarvelObject, feature_dt,
                                           gene_of_interest = NULL) {
  idx <- .build_sj_index_from_feature(MarvelObject, feature_dt)

  # keep only well-defined SJs
  idx <- idx[!is.na(strand) & !is.na(donor) & !is.na(acceptor)]
  # de-duplicate rows (OK to use unique() with by= in data.table)
  idx <- unique(idx, by = c("chrom","strand","donor","acceptor","coord.intron"))

  if (!is.null(gene_of_interest)) {
    has_gene_cols <- c("gene_short_name.start","gene_short_name.end") %in% names(idx)
    if (all(has_gene_cols)) {
      idx <- idx[(gene_short_name.start == gene_of_interest) | (gene_short_name.end == gene_of_interest)]
    }
    if (nrow(idx) == 0L) stop("No SJs found for gene: ", gene_of_interest)
  }

  setkey(idx, chrom, strand, donor, acceptor)

  trips <- idx[, {
    res <- vector("list", 0L)
    for (r in seq_len(.N)) {
      d1 <- donor[r]; a3 <- acceptor[r]; s <- strand[r]
      sj_skip <- coord.intron[r]

      a2s <- idx[donor == d1, unique(acceptor)]
      d2s <- idx[acceptor == a3, unique(donor)]
      if (!length(a2s) || !length(d2s)) next
      a2s <- a2s[!is.na(a2s)]
      d2s <- d2s[!is.na(d2s)]
      if (!length(a2s) || !length(d2s)) next

      for (a2 in a2s) for (d2 in d2s) {
        ok <- isTRUE(
          (s == "+" && d1 < a2 && a2 < d2 && d2 < a3) ||
          (s == "-" && d1 > a2 && a2 > d2 && d2 > a3)
        )
        if (!ok) next

        sj_inc1 <- idx[donor == d1 & acceptor == a2, coord.intron]
        sj_inc2 <- idx[donor == d2 & acceptor == a3, coord.intron]
        if (!length(sj_inc1) || !length(sj_inc2)) next

        res[[length(res) + 1L]] <- data.table(
          chrom   = chrom[r],
          strand  = s,
          d1 = d1, a2 = a2, d2 = d2, a3 = a3,
          sj_inc1 = sj_inc1[1],
          sj_inc2 = sj_inc2[1],
          sj_skip = sj_skip
        )
      }
    }
    if (length(res)) rbindlist(res) else NULL
  }, by = .(chrom, strand)]

  unique(trips)
}

.find_cassette_triplets_genomic <- function(MarvelObject, feature_dt,
                                            gene_of_interest = NULL) {
  feat <- .build_sj_index_from_feature(MarvelObject, feature_dt)
  # keep SJs present + clean
  feat <- feat[!is.na(left) & !is.na(right)]
  if (!is.null(gene_of_interest)) {
    if (all(c("gene_short_name.start","gene_short_name.end") %in% names(feat))) {
      feat <- feat[(gene_short_name.start == gene_of_interest) | (gene_short_name.end == gene_of_interest)]
    }
  }
  if (nrow(feat) == 0L) return(data.table())

  setkey(feat, chrom, left, right)

  trips <- feat[, {
    out <- vector("list", 0L)
    for (r in seq_len(.N)) {
      L1 <- left[r]; R3 <- right[r]; sj_skip <- coord.intron[r]
      R2s <- feat[left == L1 & right > L1 & right < R3, unique(right)]
      L2s <- feat[right == R3 & left > L1 & left < R3, unique(left)]
      if (!length(R2s) || !length(L2s)) next
      for (R2 in R2s) {
        sj_inc1 <- feat[left == L1 & right == R2, coord.intron]
        if (!length(sj_inc1)) next
        for (L2 in L2s) {
          sj_inc2 <- feat[left == L2 & right == R3, coord.intron]
          if (!length(sj_inc2)) next
          out[[length(out)+1L]] <- data.table(
            chrom = chrom[r], strand = if ("strand" %in% names(feat)) strand[r] else NA_character_,
            d1 = L1, a2 = R2, d2 = L2, a3 = R3,
            sj_inc1 = sj_inc1[1], sj_inc2 = sj_inc2[1], sj_skip = sj_skip
          )
        }
      }
    }
    if (length(out)) rbindlist(out) else NULL
  }, by = chrom]

  unique(trips)
}


psi_exon_skipping_strand <- function(MarvelObject, feature_dt,
                                     gene_of_interest = NULL,
                                     method = c("min","mean"),
                                     min_total = 0L,
                                     return_empty = TRUE,
                                     fallback_genomic = TRUE) {
  # Require data.table to be attached for `:=`
  if (!"package:data.table" %in% search()) library(data.table)

  method <- match.arg(method)

  # 1) Find cassette triplets (strand-aware), optionally fallback to genomic-order
  trips <- .find_cassette_triplets_strand(MarvelObject, feature_dt, gene_of_interest)
  if (nrow(trips) == 0L && fallback_genomic) {
    trips <- .find_cassette_triplets_genomic(MarvelObject, feature_dt, gene_of_interest)
  }

  # 2) Handle "no events" case
  if (nrow(trips) == 0L) {
    if (return_empty) {
      return(data.table(
        trip_id = integer(0), cell.id = character(0), PSI = numeric(0),
        inc = numeric(0), skip = numeric(0), tot = numeric(0),
        chrom = character(0), strand = character(0),
        d1 = integer(0), a2 = integer(0), d2 = integer(0), a3 = integer(0),
        sj_inc1 = character(0), sj_inc2 = character(0), sj_skip = character(0)
      ))
    } else {
      stop("No exon-skipping triplets detected",
           if (!is.null(gene_of_interest)) paste0(" for ", gene_of_interest), ".")
    }
  }

  # 3) De-dup + stable trip IDs
  trips <- unique(trips, by = c("chrom","strand","d1","a2","d2","a3","sj_inc1","sj_inc2","sj_skip"))
  trips[, trip_id := .I]
  setkey(trips, trip_id)

  # 4) Build long per-(SJ, cell) counts only for SJs used by the triplets
  mat <- MarvelObject$sj.count.matrix
  need <- unique(c(trips$sj_inc1, trips$sj_inc2, trips$sj_skip))
  need <- intersect(need, rownames(mat))         # safety if some SJs are missing
  if (length(need) == 0L) {
    return(data.table(
      trip_id = integer(0), cell.id = character(0), PSI = numeric(0),
      inc = numeric(0), skip = numeric(0), tot = numeric(0),
      chrom = character(0), strand = character(0),
      d1 = integer(0), a2 = integer(0), d2 = integer(0), a3 = integer(0),
      sj_inc1 = character(0), sj_inc2 = character(0), sj_skip = character(0)
    ))
  }

  X <- as.matrix(mat[need, , drop = FALSE])

  long <- as.data.table(as.table(X))
  setnames(long, c("coord.intron","cell.id","count"))
  long <- unique(long, by = c("coord.intron","cell.id"))
  setkey(long, coord.intron)

  # 5) Map triplets -> counts (allow cartesian expansion: triplet × cells)
  inc1_map <- trips[, .(trip_id, coord.intron = sj_inc1)]
  inc2_map <- trips[, .(trip_id, coord.intron = sj_inc2)]
  skip_map <- trips[, .(trip_id, coord.intron = sj_skip)]

  L1 <- long[inc1_map, on = "coord.intron", allow.cartesian = TRUE][, role := "inc1"]
  L2 <- long[inc2_map, on = "coord.intron", allow.cartesian = TRUE][, role := "inc2"]
  LS <- long[skip_map, on = "coord.intron", allow.cartesian = TRUE][, role := "skip"]

  allL <- rbindlist(list(L1, L2, LS), use.names = TRUE, fill = TRUE)

  # Drop rows with no trip_id (shouldn't happen, but be safe)
  allL <- allL[!is.na(trip_id)]

  # 6) Pivot to one row per (trip, cell)
  wide <- data.table::dcast(allL, trip_id + cell.id ~ role, value.var = "count", fill = 0)
  data.table::setDT(wide)

  # Ensure columns exist even if a role is globally missing
  if (!"inc1" %in% names(wide)) wide[, inc1 := 0]
  if (!"inc2" %in% names(wide)) wide[, inc2 := 0]
  if (!"skip" %in% names(wide)) wide[, skip := 0]

  # 7) Compute PSI
  if (method == "min") {
    wide[, inc := pmin(inc1, inc2)]
  } else {
    wide[, inc := (inc1 + inc2) / 2]
  }
  wide[, tot := inc + skip]
  if (min_total > 0L) wide[tot < min_total, `:=`(inc = NA_real_, tot = NA_real_)]

  wide[, PSI := fifelse(tot > 0, (inc / tot) * 100, NA_real_)]

  # 8) Attach triplet annotations
  out <- merge(
    wide,
    trips[, .(trip_id, chrom, strand, d1, a2, d2, a3, sj_inc1, sj_inc2, sj_skip)],
    by = "trip_id",
    all.x = TRUE
  )

  # Consistent column order
  data.table::setcolorder(out, c("trip_id","cell.id","PSI","inc","skip","tot",
                                 "chrom","strand","d1","a2","d2","a3",
                                 "sj_inc1","sj_inc2","sj_skip"))
  out[]
}

# Updated: now returns donor/acceptor (and chrom/strand) for donor|acceptor metrics,
# and event coordinates for exon_skip. Keeps your old columns too.
make_psi_long_new <- function(MarvelObject,
                              feature_dt,
                              gene_of_interest,
                              cis_gene = gene_of_interest,
                              type = c("donor","acceptor","exon_skip"),
                              min_cell_total = 1,
                              min_total_exon = 2) {
  type <- match.arg(type)

  if (type %in% c("donor","acceptor")) {
    if (type == "donor") {
      dt <- psi_donor_usage_strand(MarvelObject, feature_dt,
                                   gene_of_interest = gene_of_interest,
                                   min_cell_total = min_cell_total)
    } else {
      dt <- psi_acceptor_usage_strand(MarvelObject, feature_dt,
                                      gene_of_interest = gene_of_interest,
                                      min_cell_total = min_cell_total)
    }
    # Standard SJ label = the actual junction rowname
    dt[, SJ := coord.intron]

    # Attach sample metadata
    smeta <- as.data.table(MarvelObject$sample.metadata)
    keep_cols <- intersect(c("cell.id","guide_crispr","cell_line","gene"), names(smeta))
    dt <- dt[smeta[, ..keep_cols], on = "cell.id"]

    # Restrict to cis+ntc if 'gene' exists
    if ("gene" %in% names(dt)) {
      dt <- dt[gene %in% c(cis_gene, "ntc")]
    }

    # Return with donor/acceptor & chrom/strand carried along
    out <- dt[, .(SJ, cell.id, PSI, guide_crispr, cell_line, gene,
                  chrom, strand, donor, acceptor)]
    return(out[!is.na(PSI)])
  }

  # exon_skip
  dt <- psi_exon_skipping_strand(MarvelObject, feature_dt,
                                 gene_of_interest = gene_of_interest,
                                 method = "min",
                                 min_total = min_total_exon)
  # Human-readable event label; keep a stable ID too
  dt[, `:=`(
    event_id = sprintf("ES#%d", trip_id),
    SJ = sprintf("ES#%d %s%s d1=%d a2=%d d2=%d a3=%d",
                 trip_id, chrom, strand, d1, a2, d2, a3)
  )]

  smeta <- as.data.table(MarvelObject$sample.metadata)
  keep_cols <- intersect(c("cell.id","guide_crispr","cell_line","gene"), names(smeta))
  dt <- dt[smeta[, ..keep_cols], on = "cell.id"]
  if ("gene" %in% names(dt)) {
    dt <- dt[gene %in% c(cis_gene, "ntc")]
  }

  # For ES, include the event coordinates
  out <- dt[, .(SJ, cell.id, PSI, guide_crispr, cell_line, gene,
                chrom, strand, d1, a2, d2, a3,
                sj_inc1, sj_inc2, sj_skip, event_id)]
  out[!is.na(PSI)]
}

# Keep SJs where at least `minor_prop` of cells are NOT in the majority PSI bin
# - bin: PSI bin width in %. Example: 1 (=round to 1%), 5 (to nearest 5%), etc.
# - min_cells: require at least this many non-NA cells for the SJ to be considered
filter_sjs_by_minor_fraction <- function(psi_long,
                                         minor_prop = 0.05,
                                         bin = 1,
                                         min_cells = 10) {
  stopifnot(bin > 0)
  dt <- psi_long[!is.na(PSI), .(SJ, PSI)]

  # Bin PSI (e.g., 99.7 -> 100 if bin=1; 97.4 -> 95 if bin=5)
  dt[, psi_bin := round(PSI / bin) * bin]

  # Count per bin per SJ
  freq <- dt[, .N, by = .(SJ, psi_bin)]

  # Majority bin size per SJ
  maj <- freq[freq[, .I[which.max(N)], by = SJ]$V1][, .(SJ, N_major = N)]

  # Total usable cells per SJ
  tot <- dt[, .N, by = SJ][, .(SJ, N_total = N)]

  # Merge + compute minority fraction
  agg <- maj[tot, on = "SJ"]
  agg[, minor_fraction := 1 - (N_major / N_total)]

  # Keep SJs meeting both coverage and heterogeneity
  keep_sj <- agg[N_total >= min_cells & minor_fraction >= minor_prop, SJ]

  psi_long[SJ %in% keep_sj]
}
