/**
 *
 */
package org.theseed.dl4j;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.stream.IntStream;

import org.apache.commons.math3.stat.regression.SimpleRegression;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.train.ITrainingProcessor;
import org.theseed.dl4j.train.ModelType;
import org.theseed.io.TabbedLineReader;
import org.theseed.utils.BaseProcessor;
import org.theseed.utils.ParseFailureException;

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
public class PearsonProcessor extends BaseProcessor {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(PearsonProcessor.class);
    /** output stream */
    private OutputStream outStream;
    /** flags indicating input columnn indices */
    private BitSet inCols;
    /** input training file */
    private TabbedLineReader trainStream;
    /** output column index */
    private int outColIdx;
    /** array of input column names */
    private String[] headers;

    // COMMAND-LINE OPTIONS

    /** output file (if not STDOUT) */
    @Option(name = "-o", aliases = { "--output" }, metaVar = "outFile.tbl", usage = "output file (if not STDOUT)")
    private File outFile;

    /** model type */
    @Option(name = "-t", aliases = { "--type" }, usage = "model directory type")
    private ModelType type;

    /** input model directory */
    @Argument(index = 0, metaVar = "modelDir", usage = "input model directory")
    private File modelDir;

    /** output column name */
    @Argument(index = 1, metaVar = "outCol", usage = "output column name")
    private String outCol;

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

    @Override
    protected void setDefaults() {
        this.outFile = null;
    }

    @Override
    protected boolean validateParms() throws IOException, ParseFailureException {
        // Validate the model directory.
        if (! this.modelDir.isDirectory())
            throw new FileNotFoundException("Model directory " + this.modelDir + " is not found or invalid.");
        // Load the model.
        ITrainingProcessor processor = ModelType.create(this.type);
        if (! processor.initializeForPredictions(this.modelDir))
            throw new ParseFailureException("Parameter error:  model type is probably wrong.");
        // Now we need the list of input columns.  This requires opening the input file,
        // So we need protection.
        File trainFile = new File(this.modelDir, "training.tbl");
        this.trainStream = new TabbedLineReader(trainFile);
        boolean retVal = false;
        try {
            // Compute the output column index.
            this.outColIdx = this.trainStream.findField(this.outCol);
            this.headers = this.trainStream.getLabels();
            this.inCols = new BitSet(headers.length);
            Set<String> skipSet = new HashSet<String>(headers.length);
            skipSet.addAll(processor.getLabelCols());
            skipSet.addAll(processor.getMetaList());
            // Test all the column labels.
            int count = 0;
            for (int i = 0; i < this.headers.length; i++) {
                if (! skipSet.contains(this.headers[i]) && i != this.outColIdx) {
                    this.inCols.set(i);
                    count++;
                }
            }
            log.info("{} input columns found in training file.", count);
            // Connect to the output stream.
            if (this.outFile == null) {
                log.info("Output will be to STDOUT.");
                this.outStream = System.out;
            } else {
                log.info("Output will be to {}.", this.outFile);
                this.outStream = new FileOutputStream(this.outFile);
            }
            retVal = true;
        } finally {
            // If we are failing, close the input file.
            if (! retVal)
                this.trainStream.close();
        }
        return true;
    }

    @Override
    protected void runCommand() throws Exception {
        try {
            // Create a regression object for each input column.
            SimpleRegression[] computers = IntStream.range(0, this.trainStream.size()).mapToObj(i -> new SimpleRegression())
                    .toArray(SimpleRegression[]::new);
            // Read the input file.
            log.info("Processing training file.");
            for (TabbedLineReader.Line line : this.trainStream) {
                // Get the output column.
                String outString = line.get(this.outColIdx);
                if (! outString.isEmpty()) {
                    double outVal = Double.valueOf(outString);
                    for (int i = 0; i < computers.length; i++) {
                        if (this.inCols.get(i)) {
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
                if (this.inCols.get(i)) {
                    String name = this.headers[i];
                    double pc = computers[i].getR();
                    corrs.add(new Correlation(name, pc));
                }
            }
            // Now we write the output.
            log.info("Producing output.");
            try (PrintWriter writer = new PrintWriter(this.outStream)) {
                writer.println("col_name\tcorrelation");
                for (Correlation corr : corrs)
                    writer.format("%s\t%6.4f%n", corr.getColName(), corr.getCorrelation());
            }
        } finally {
            // Insure the files are closed.
            if (this.outFile == null)
                this.outStream.close();
            this.trainStream.close();
        }
    }

}
