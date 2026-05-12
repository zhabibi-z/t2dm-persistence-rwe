#!/usr/bin/env perl
# report_formatter.pl — converts pipeline CSV outputs to a formatted text summary report.
# Usage: perl scripts/report_formatter.pl outputs/tables/ > outputs/study_report.txt

use strict;
use warnings;
use File::Basename;
use POSIX qw(strftime);

my $table_dir = $ARGV[0] // "outputs/tables";
my $timestamp = strftime("%Y-%m-%d %H:%M:%S", localtime);

print "=" x 80 . "\n";
print "T2DM TREATMENT PERSISTENCE RWE — STUDY RESULTS SUMMARY\n";
print "Generated: $timestamp\n";
print "=" x 80 . "\n\n";

sub print_csv_table {
    my ($filepath, $title) = @_;
    return unless -f $filepath;

    print "-" x 60 . "\n";
    print "  $title\n";
    print "-" x 60 . "\n";

    open(my $fh, '<', $filepath) or do {
        print "  [Could not read $filepath]\n\n";
        return;
    };

    my $header = <$fh>;
    chomp $header;
    my @cols = split(/,/, $header);

    # Print header row
    printf("  %-30s", shift @cols);
    printf(" %15s", $_) for @cols;
    print "\n";
    print "  " . "-" x 58 . "\n";

    while (my $line = <$fh>) {
        chomp $line;
        my @vals = split(/,/, $line);
        printf("  %-30s", shift @vals);
        for my $v (@vals) {
            # Round numeric values to 4 decimal places
            if ($v =~ /^-?\d+\.?\d*$/) {
                printf(" %15.4f", $v);
            } else {
                printf(" %15s", $v);
            }
        }
        print "\n";
    }
    close $fh;
    print "\n";
}

# Section 1: Cohort summary
print_csv_table("$table_dir/cohort_summary.csv", "1. Cohort Characteristics");

# Section 2: Comorbidity prevalence
print_csv_table("$table_dir/comorbidity_prevalence.csv", "2. Comorbidity Prevalence by Drug Class");

# Section 3: TTD summary statistics
print_csv_table("$table_dir/ttd_summary.csv", "3. Time-to-Discontinuation Summary");

# Section 4: Cox model results
print_csv_table("$table_dir/cox_ttd_results.csv", "4. Cox PH Model — TTD Hazard Ratios");

# Section 5: Time-varying Cox
print_csv_table("$table_dir/cox_timevarying_results.csv", "5. Time-Varying Cox — Comorbidity Onset Effects");

# Section 6: TTC Cox
print_csv_table("$table_dir/cox_ttc_results.csv", "6. TTC Cox — Hazard Ratios for Comorbidity Onset");

# Section 7: Pearson correlations
print_csv_table("$table_dir/correlations.csv", "7. Pearson Correlation: Comorbidity x TTD");

# Section 8: Hypothesis tests
print_csv_table("$table_dir/kruskal_results.csv", "8. Kruskal-Wallis + Dunn Post-hoc (BH-FDR)");

# Section 9: ML performance
print_csv_table("$table_dir/ml_cv_results.csv", "9. XGBoost 5-Fold CV Performance");

print "=" x 80 . "\n";
print "END OF REPORT\n";
print "=" x 80 . "\n";
