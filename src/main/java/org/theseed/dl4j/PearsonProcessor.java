/**
 *
 */
package org.theseed.dl4j;

import java.io.PrintWriter;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.IntStream;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.theseed.dl4j.train.TrainingAnalysisProcessor;
import org.theseed.io.TabbedLineReader;

/**
 * This command computes the pearson coefficients for each of the input columns of a regression model and a selected
 * output column.
 *
 * The positional parameters are the name of the model directory and and the name of the output column.  The
 * command-line options are as follows.
 *
 * -h	display command-line usage
 * -v	display more frequent log messages
 * -o	output file, if not STDOUT
 * -t	model type; default REGRESSION
 *
 * @author Bruce Parrello
 *
 */
public class PearsonProcessor extends TrainingAnalysisProcessor {

     /**
     * This class is used to store the correlation value for each input column.  It is
     * sorted by highest absolute correlation value, then column name.
     */
    protected static class Correlation implements Comparable<Correlation> {

        private String colName;
        private double correlation;

        /**
         * Construct a correlation.
         *
         * @param col	name of the correlated column
         * @param pc	Pearson coefficient
         */
        public Correlation(String col, double pc) {
            this.colName = col;
            this.correlation = pc;
        }

        @Override
        public int compareTo(Correlation o) {
            int retVal = Double.compare(Math.abs(o.correlation), Math.abs(this.correlation));
            if (retVal == 0)
                retVal = this.colName.compareTo(o.colName);
            return retVal;
        }

        /**
         * @return the column name
         */
        public String getColName() {
            return this.colName;
        }

        /**
         * @return the correlation
         */
        public double getCorrelation() {
            return this.correlation;
        }

    }

    /**
     * Process the training file to produce the correlations.
     */
    public void processCommand() {
        // Create a regression object for each input column.
        SimpleRegression[] computers = IntStream.range(0, this.getTrainStream().size()).mapToObj(i -> new SimpleRegression())
                .toArray(SimpleRegression[]::new);
        // Read the input file.
        log.info("Processing training file.");
        for (TabbedLineReader.Line line : this.getTrainStream()) {
            // Get the output column.
            String outString = line.get(this.getOutColIdx());
            if (! outString.isEmpty()) {
                double outVal = Double.valueOf(outString);
                for (int i = 0; i < computers.length; i++) {
                    if (this.getInCols(i)) {
                        double iVal = line.getDouble(i);
                        computers[i].addData(iVal, outVal);
                    }
                }
            }
        }
        log.info("Computing correlations.");
        // We use a tree set to get the output in the correct order.
        SortedSet<Correlation> corrs = new TreeSet<Correlation>();
        for (int i = 0; i < computers.length; i++) {
            if (this.getInCols(i)) {
                String name = this.getHeader(i);
                double pc = computers[i].getR();
                corrs.add(new Correlation(name, pc));
            }
        }
        // Now we write the output.
        log.info("Producing output.");
        try (PrintWriter writer = new PrintWriter(this.getOutStream())) {
            writer.println("col_name\tcorrelation");
            for (Correlation corr : corrs)
                writer.format("%s\t%6.4f%n", corr.getColName(), corr.getCorrelation());
        }
    }

    @Override
    protected void setCommandDefaults() {
    }

}
