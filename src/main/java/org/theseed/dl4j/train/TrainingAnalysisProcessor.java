/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.BitSet;
import java.util.HashSet;
import java.util.Set;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.PearsonProcessor;
import org.theseed.io.TabbedLineReader;
import org.theseed.utils.BaseProcessor;
import org.theseed.utils.ParseFailureException;

/**
 * This is a base class for commands that need to process the input columns of a training set.  The client
 * gets access to an open training set input stream and a list of the input columns.
 *
 * @author Bruce Parrello
 *
 */
public abstract class TrainingAnalysisProcessor extends BaseProcessor {

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

   public TrainingAnalysisProcessor() {
        super();
    }

    @Override
    protected final void setDefaults() {
        this.outFile = null;
        this.setCommandDefaults();
    }

    /**
     * Specify the default values for subclass options.
     */
    protected abstract void setCommandDefaults();

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
            this.outColIdx = this.getTrainStream().findField(this.outCol);
            this.headers = this.getTrainStream().getLabels();
            this.inCols = new BitSet(headers.length);
            Set<String> skipSet = new HashSet<String>(headers.length);
            skipSet.addAll(processor.getLabelCols());
            skipSet.addAll(processor.getMetaList());
            // Test all the column labels.
            int count = 0;
            for (int i = 0; i < this.headers.length; i++) {
                if (! skipSet.contains(this.headers[i]) && i != this.getOutColIdx()) {
                    this.inCols.set(i);
                    count++;
                }
            }
            log.info("{} input columns found in training file.", count);
            // Connect to the output stream.
            if (this.getOutFile() == null) {
                log.info("Output will be to STDOUT.");
                this.outStream = System.out;
            } else {
                log.info("Output will be to {}.", this.getOutFile());
                this.outStream = new FileOutputStream(this.getOutFile());
            }
            retVal = true;
        } finally {
            // If we are failing, close the input file.
            if (! retVal)
                this.getTrainStream().close();
        }
        return true;
    }

    @Override
    protected final void runCommand() throws Exception {
        try {
            this.processCommand();
        } finally {
            // Insure the files are closed.
            if (this.getOutFile() == null)
                this.getOutStream().close();
            this.getTrainStream().close();
        }
    }

    /**
     * Process the training file to produce output.
     */
    protected abstract void processCommand();

    /**
     * @return the training file input stream
     */
    public TabbedLineReader getTrainStream() {
        return trainStream;
    }

    /**
     * @return the output column index
     */
    public int getOutColIdx() {
        return outColIdx;
    }

    /**
     * @return TRUE if the indicated column is an input column, else FALSE
     *
     * @param idx		index of the column to check
     */
    public boolean getInCols(int idx) {
        return inCols.get(idx);
    }

    /**
     * @return the output Stream
     */
    public OutputStream getOutStream() {
        return outStream;
    }

    /**
     * @return the specified header
     *
     * @param idx	index of the header desired
     */
    public String getHeader(int idx) {
        return this.headers[idx];
    }

    /**
     * @return the outFile
     */
    public File getOutFile() {
        return outFile;
    }

}
