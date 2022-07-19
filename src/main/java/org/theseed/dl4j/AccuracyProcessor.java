/**
 *
 */
package org.theseed.dl4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.io.TabbedLineReader;
import org.theseed.utils.BaseReportProcessor;
import org.theseed.utils.ParseFailureException;

/**
 * This command analyzes predictions to determine a relationship between the testing of on/off columns and
 * prediction accuracy.  We read the training/testing set to determine the number of on-instances used to
 * build the model, and we read a validation set to determine the accuracy.
 *
 * The positional parameters are the name of the training/testing file, the name of a prediction input file, the
 * name of the file containing the predictions, the name of the ID column, and the name of the output column
 * containing actual values.  Note that the ID column name must be the same in both prediction files.
 *
 * The report is written to the standard output.
 *
 * The command-line options are as follows.
 *
 * -h	display command-line usage
 * -v	display more frequent log messages
 * -o	output file for report (if not STDOUT)
 *
 * --pCol		index (1-based) or name of the input file column containing the predictions
 * --cutoff		cutoff for psuedo-accuracy computation
 *
 *
 * @author Bruce Parrello
 *
 */
public class AccuracyProcessor extends BaseReportProcessor {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(AccuracyProcessor.class);
    /** map of column names to accuracy items */
    private Map<String, AccuracyItem> columnMap;
    /** set of column names to ignore */
    private Set<String> ignoreColumns;

    // COMMAND-LINE OPTIONS

    /** index or name of the input file column containing the predictions */
    @Option(name = "--pCol", aliases = { "--col" }, usage = "index (1-based) or name of the input column containing the predictions")
    private String pColName;

    /** cutoff for computing pseudo-accuracy */
    @Option(name = "--cutoff", metaVar = "2.0", usage = " cutoff for converting output values to high/low classes")
    private double cutoff;

    /** training/testing file */
    @Argument(index = 0, metaVar = "training.tbl", usage = "name of the training/testing file for the model", required = true)
    private File trainFile;

    /** prediction input file */
    @Argument(index = 1, metaVar = "validation.tbl", usage = "name of the prediction input file for validation", required = true)
    private File predictInFile;

    /** prediction output file */
    @Argument(index = 2, metaVar = "predictions.tbl", usage = "name of the prediction output file for validation", required = true)
    private File predictOutFile;

    /** ID column name */
    @Argument(index = 3, metaVar = "idCol", usage = "name of column containing row IDs", required = true)
    private String idColName;

    /** output column name */
    @Argument(index = 4, metaVar = "valueCol", usage = "name of column containing actual output value", required = true)
    private String outColName;

    @Override
    protected void setReporterDefaults() {
        this.pColName = "predicted";
        this.cutoff = 1.2;
    }

    @Override
    protected void validateReporterParms() throws IOException, ParseFailureException {
        // Validate the files.
        if (! this.trainFile.canRead())
            throw new FileNotFoundException("Training/testing file " + this.trainFile + " is not found or unreadable.");
        if (! this.predictOutFile.canRead())
            throw new FileNotFoundException("Prediction file " + this.predictOutFile + " is not found or unreadable.");
        if (! this.predictInFile.canRead())
            throw new FileNotFoundException("Prediction file " + this.predictInFile + " is not found or unreadable.");
        // Set up the ignore-column set.
        this.ignoreColumns = Set.of(this.pColName, this.outColName, this.idColName);
        // Log the pseudo-accuracy cutoff.
        log.info("Pseudo-accuracy cutoff is {}.", this.cutoff);
        AccuracyItem.setCutoff(this.cutoff);
    }

    @Override
    protected void runReporter(PrintWriter writer) throws Exception {
        // Build the accuracy-item map from the training/testing file.  This nets us the row counts.
        this.readTrainingFile();
        // Compute the accuracy from the predictions.
        this.readPredictionFile();
        // Sort the accuracy items.
        var items = new ArrayList<AccuracyItem>(this.columnMap.size());
        this.columnMap.values().stream().filter(x -> x.isValid() && x.getRowCount() > 0).forEach(x -> items.add(x));
        Collections.sort(items);
        Set<String> validSet = items.stream().map(x -> x.getColName()).collect(Collectors.toSet());
        // Write the report.
        writer.println("label\tcount\tpred_count\twidth\tbreadth\tMAE\tF1\taccuracy");
        for (var item : items) {
            if (item.isValid()) {
                var label = item.getColName();
                if (item.getPredCount() > 0) {
                    writer.format("%s\t%d\t%d\t%d\t%8.3f\t%8.3f\t%8.3f\t%8.3f%n", label, item.getRowCount(), item.getPredCount(),
                            item.getWidth(validSet), item.getBreadth(validSet), item.getMAE(), item.getF1(),
                            item.getAccuracy());
                }
            }
        }
    }

    /**
     * Read the training file, and create the accuracy map.  At the end of this method, the accuracy map
     * will contain the row counts used to determine how much input we have on each feature.
     *
     * @throws IOException
     */
    private void readTrainingFile() throws IOException {
        log.info("Processing training file {}.", this.trainFile);
        // Open the training file for input.
        try (var trainStream = new TabbedLineReader(this.trainFile)) {
            // Get the array of column labels.
            var labels = trainStream.getLabels();
            // Create the accuracy map.  We use a tree map so that the items are sorted.  The number of items is
            // expected to be small.
            this.columnMap = new HashMap<String, AccuracyItem>(labels.length * 4 / 3 + 1);
            // Use the label array to create the initial items.  We keep them in a list so they can be processed
            // easily in column order.
            var columns = new ArrayList<AccuracyItem>(labels.length);
            int idx = 0;
            for (String label : labels) {
                // Create the accuracy item.  If it is one of the ignore-columns (predicted-value or actual-value),
                // we mark it invalid immediately.
                var columnItem = new AccuracyItem(label, idx, ! this.ignoreColumns.contains(label));
                this.columnMap.put(label, columnItem);
                columns.add(columnItem);
                // Update the index for the next column.
                idx++;
            }
            log.info("{} columns found in training file.", this.columnMap.size());
            // Now we must loop through the file, counting rows.
            for (var line : trainStream) {
                // For this row, compute the columns turned on.
                var onCols = new TreeSet<String>();
                for (var item : columns) {
                    if (item.isValid()) {
                        double val = line.getDoubleSafe(item.getColIdx());
                        if (! Double.isFinite(val))
                            item.invalidate();
                        else if (val == 1.0)
                            onCols.add(item.getColName());
                    }
                }
                // Now record this row for each feature turned on.
                for (var colName : onCols) {
                    var item = this.columnMap.get(colName);
                    item.recordRow(onCols);
                }
            }
            // Determine the number of useful columns.
            long useful = columns.stream().filter(x -> x.isValid()).count();
            log.info("{} columns are of the on/off type.", useful);
        }
    }

    /**
     * Read the prediction files and tally the prediction accuracy for each on/off column.  Essentially,
     * for each row we examine every column marked as VALID in the training file.  For the columns where
     * the feature is turned on in the row, we record the prediction error.
     *
     * @throws IOException
     */
    private void readPredictionFile() throws IOException {
        // Get a hash of sample IDs to prediction values.
        log.info("Processing prediction output file {}.");
        Map<String, Double> predictMap = new HashMap<String, Double>(1000);
        try (var predictStream = new TabbedLineReader(this.predictOutFile)) {
            int idIdx = predictStream.findField(this.idColName);
            int predictIdx = predictStream.findField(this.pColName);
            for (var line : predictStream) {
                double prediction = line.getDouble(predictIdx);
                predictMap.put(line.get(idIdx), prediction);
            }
            log.info("{} predictions found.", predictMap.size());
        }
        // Open the prediction file.
        log.info("Processing prediction input file {}.", this.predictInFile);
        try (var predictStream = new TabbedLineReader(this.predictInFile)) {
            // Get the column label array.
            var labels = predictStream.getLabels();
            // Compute the sample-id and actual-value columns.  If either name is invalid, we
            // will get an IOException.
            var actualIdx = predictStream.findField(this.outColName);
            var idIdx = predictStream.findField(this.idColName);
            // Insure these columns are marked invalid.  It is likely, but if the user specified a column
            // index, the check in readTrainingFile won't catch the columns.
            this.invalidateItem(labels[actualIdx]);
            this.invalidateItem(labels[idIdx]);
            // Update the column indices.
            IntStream.range(0, labels.length).filter(i -> this.columnMap.containsKey(labels[i]))
                    .forEach(i -> this.columnMap.get(labels[i]).setColIdx(i));
            // Now we loop through the records.
            int count = 0;
            int errCount = 0;
            int notFound = 0;
            for (var line : predictStream) {
                // Compute the prediction error.
                String id = line.get(idIdx);
                if (! predictMap.containsKey(id)) {
                    log.warn("Row ID \"{}\" not found in prediction output.", id);
                    notFound++;
                } else {
                    double actual = line.getDouble(actualIdx);
                    double predicted = predictMap.get(id);
                    // Process the columns for this record.
                    for (var label : labels) {
                        var item = this.columnMap.get(label);
                        if (item != null && item.checkRow(line)) {
                            item.recordError(actual, predicted);
                            errCount++;
                        }
                    }
                    count++;
                }
            }
            log.info("{} rows read from prediction file.  {} prediction values missing. {} error values recorded.",
                    count, notFound, errCount);
        }

    }

    /**
     * Insure the column with the specified name (if it exists) is marked invalid.
     *
     * @param label		column name
     */
    private void invalidateItem(String label) {
        var item = this.columnMap.get(label);
        if (item != null)
            item.invalidate();
    }

}
