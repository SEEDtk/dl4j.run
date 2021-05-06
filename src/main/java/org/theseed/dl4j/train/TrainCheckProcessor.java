/**
 *
 */
package org.theseed.dl4j.train;

import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.kohsuke.args4j.Option;
import org.theseed.counters.CountMap;
import org.theseed.dl4j.CrossReference;
import org.theseed.io.TabbedLineReader;

/**
 * This command is used for training files where the values are discreet (a common case).  For each value in each column,
 * the number of times it occurs with various values in the other columns are displayed.
 *
 * The positional parameters are the name of the model directory and the name of the output column.
 *
 * The command-line options are as follows.
 *
 * -h	display command-line usage
 * -v	display more frequent log messages
 * -t	model type; default REGRESSION
 * -o	output file (if not STDOUT)
 *
 * --nonzero	if specified, column values of 0 will be ignored
 *
 * @author Bruce Parrello
 *
 */
public class TrainCheckProcessor extends TrainingAnalysisProcessor {

    // FIELDS
    /** count of each cross-reference */
    private CountMap<CrossReference> crossCounts;
    /** statistics for each cross-reference */
    private Map<CrossReference, SummaryStatistics> crossStats;

    // COMMAND-LINE OPTIONS

    @Option(name = "--nonzero", usage = "if specified, zero values in a column will be ignored")
    private boolean nonZero;

    @Override
    protected void setCommandDefaults() {
        this.nonZero = false;
    }


    @Override
    protected void processCommand() {
        // Compute the number of columns to process.
        int n = this.getTrainStream().size();
        // Create the cross-reference maps.
        this.crossCounts = new CountMap<CrossReference>();
        this.crossStats = new HashMap<CrossReference, SummaryStatistics>(n*n);
        // We process one input record at a time, generating all the cross-references.
        // Loop through the input.
        int count = 0;
        for (TabbedLineReader.Line line : this.getTrainStream()) {
            count++;
            log.info("Processing line {}.", count);
            // Loop through the columns.
            for (int i = 0; i < n; i++) {
                if (this.getInCols(i)) {
                    // Here we have a real input column.
                    String col1 = this.getHeader(i);
                    double val1 = line.getDouble(i);
                    if (! this.nonZero || val1 != 0.0) {
                        for (int j = i + 1; j < n; j++) {
                            if (this.getInCols(j)) {
                                // Here we have the second column.
                                String col2 = this.getHeader(j);
                                double val2 = line.getDouble(j);
                                if (! this.nonZero || val2 != 0.0) {
                                    CrossReference ref = new CrossReference(col1, val1, col2, val2);
                                    this.crossCounts.count(ref);
                                    SummaryStatistics stats = this.crossStats.computeIfAbsent(ref, x -> new SummaryStatistics());
                                    stats.addValue(line.getDouble(this.getOutColIdx()));
                                }
                            }
                        }
                    }
                }
            }
        }
        log.info("{} cross-references found in {} lines.", this.crossCounts.size(), count);
        try (PrintWriter writer = new PrintWriter(this.getOutStream())) {
            writer.println("col_1\tcol_1_val\tcol_2\tcol_2_val\tcount\tmean_out\tstdev_out");
            for (CrossReference ref : this.crossCounts.keys()) {
                int counter = this.crossCounts.getCount(ref);
                SummaryStatistics stats = this.crossStats.get(ref);
                writer.format("%s\t%d\t%8.4f\t%8.4f%n", ref.toString(), counter, stats.getMean(), stats.getStandardDeviation());
            }
        }
    }

}
