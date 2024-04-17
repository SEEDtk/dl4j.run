package org.theseed.dl4j.train;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.basic.BaseReportProcessor;
import org.theseed.basic.ParseFailureException;
import org.theseed.dl4j.RocItem;
import org.theseed.io.TabbedLineReader;

/**
 * This command takes as input a regression prediction file and produces an output plot for ROC.
 * The positional parameters are the name of the predicted label.  It is expected that the actual
 * value will be in an input column with the label name and the predicted value in a column with
 * the name "o-" followed by the label (e.g. "production" and "o-production".
 *
 * The positional parameter is the name of the label column.  The prediction file comes in on
 * the standard input, and the report is produced on the standard output.
 *
 * The command-line options are as follows.
 *
 * -h	display command-line usage
 * -v	display more frequent log messages
 * -o	output file (if not STDOUT)
 * -i	input file (if not STDIN)
 *
 * @author Bruce Parrello
 *
 */
public class RocProcessor extends BaseReportProcessor {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(RocProcessor.class);
    /** list of predicted/actual pairs */
    private List<RocItem> records;
    /** input stream */
    private TabbedLineReader inStream;
    /** index of predicted column */
    private int predictedCol;
    /** index of actual column */
    private int actualCol;

    // COMMAND-LINE OPTIONS

    @Option(name = "input", aliases = { "-i" }, metaVar = "predictions.tbl", usage = "input file (if not STDIN)")
    private File inFile;

    @Argument(index = 0, metaVar = "label", usage = "label to use for predicted/actual")
    private String inLabel;

    @Override
    protected void setReporterDefaults() {
        this.inFile = null;
    }

    @Override
    protected void validateReporterParms() throws IOException, ParseFailureException {
        if (this.inFile == null) {
            log.info("Input data will be taken from standard input.");
            this.inStream = new TabbedLineReader(System.in);
        } else {
            log.info("Input data will be taken from {}.", this.inFile);
            this.inStream = new TabbedLineReader(this.inFile);
        }
        this.actualCol = this.inStream.findField(this.inLabel);
        this.predictedCol = this.inStream.findField("o-" + this.inLabel);
    }

    @Override
    protected void runReporter(PrintWriter writer) throws Exception {
        // First, we spool the input file into our array.
        this.records = new ArrayList<RocItem>(1000);
        for (TabbedLineReader.Line line : this.inStream) {
            RocItem newItem = new RocItem(line.getDouble(this.predictedCol),
                    line.getDouble(this.actualCol));
            this.records.add(newItem);
        }
        // Now sort it by prediction.
        Collections.sort(this.records, new RocItem.ByPredicted());
        log.info("{} input records found.", this.records.size());
        // Initialize the output.
        writer.println("Threshold\tTP\tTN\tFP\tFN\tTPR\tFPR\tAccuracy");
        // We loop through the possible thresholds, producing output.  We need an extra
        // variable to eliminate duplicates.
        double oldThreshold = Double.POSITIVE_INFINITY;
        for (RocItem item : this.records) {
            double newThreshold = item.getPredicted();
            if (newThreshold < oldThreshold) {
                // Here we have a new possible threshold, so run with it.
                this.processThreshold(writer, newThreshold);
                oldThreshold = newThreshold;
            }
        }
    }

    /**
     * Produce an output line for the specified threshold.
     *
     * @param writer
     * @param threshold
     */
    private void processThreshold(PrintWriter writer, double threshold) {
        // This is the confusion matrix.  The first index is actual, the second is predicted.
        int[][] matrix = new int[2][2];
        for (RocItem item : this.records) {
            int actual = (item.getActual() >= threshold ? 1 : 0);
            int predicted = (item.getPredicted() >= threshold ? 1 : 0);
            matrix[actual][predicted]++;
        }
        // Now we compute our metrics.
        double accuracy = (matrix[0][0] + matrix[1][1]) / (double) this.records.size();
        double fpr = 0.0;
        if (matrix[0][1] > 0)
            fpr = matrix[0][1] / (double) (matrix[0][1] + matrix[0][0]);
        double tpr = 0.0;
        if (matrix[1][1] > 0)
            tpr = matrix[1][1] / (double) (matrix[1][1] + matrix[1][0]);
        // Finally, write the output.
        writer.format("%8.4f\t%d\t%d\t%d\t%d\t%6.4f\t%6.4f\t%6.4f%n",
                threshold, matrix[1][1], matrix[0][0], matrix[0][1], matrix[1][0], tpr, fpr, accuracy);
    }

}
