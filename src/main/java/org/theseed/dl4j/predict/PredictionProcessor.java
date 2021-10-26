/**
 *
 */
package org.theseed.dl4j.predict;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.ChannelDataSetReader;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.utils.ICommand;

/**
 * This method uses an existing model to make predictions.  It takes as input a tab-delimited file of feature
 * data and metadata, and outputs each metadata column with the predictions.
 *
 * The first positional parameter is the name of the model directory.  This contains a "model.ser" file with
 * the model and normalizer in it as well as a "labels.txt" file that contains the classification labels, in
 * order.
 *
 * If the model directory also contains a "channels.tbl" file, then the input will be processed in
 * channel mode.  The aforementioned file must be tab-delimited with headers.  The first column of
 * each record should be a string, and the remaining columns should be a vector of floating-point
 * numbers.  In this case, the input is considered to be two-dimensional.  Each input string is
 * replaced by the corresponding vector.  The result can be used as a kind of one-hot representation
 * of the various strings, but it can be more complicated.  For example, if the input is DNA nucleotides,
 * an ambiguity code would contain fractional numbers in multiple positions of the vector.
 *
 * The predictions will appear on the standard output, but unless the logback.xml file is altered, the standard
 * error output will contain trace messages.  Therefore, redirection is required on the command line.
 *
 * The command-line options are as follows.
 *
 * -i	the name of the input file of predictions; the default is the standard input
 * -c	heading to put on the result column; the default is "predicted"
 * -o 	output file (if not STDOUT)
 *
 * --meta			a comma-delimited list of the metadata columns; these columns are ignored during training; the
 * 					default is none
 * --name			the model file name (the default is "model.ser" in the model directory)
 * --regression		if specified, all confidences are output rather than the label with the highest confidence;
 * 					this is recommended for regression models

 * @author Bruce Parrello
 *
 */
public class PredictionProcessor implements ICommand {

    // FIELDS
    /** list of metadata column labels */
    private List<String> metaList;
    /** array of labels */
    private List<String> labels;
    /** input dataset reader */
    private TabbedDataSetReader reader;
    /** model to use for predictions */
    private MultiLayerNetwork model;
    /** output print writer */
    private PrintStream writer;

    /** logging facility */
    private static Logger log = LoggerFactory.getLogger(PredictionProcessor.class);

    // COMMAND LINE

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    /** input prediction set */
    @Option(name="-i", aliases={"--input"}, metaVar="requests.tbl",
            usage="input prediction set file")
    private File inputFile;

    /** comma-delimited list of metadata column names */
    @Option(name="--meta", metaVar="name,date", usage="comma-delimited list of metadata columns")
    private String metaCols;

    /** heading to put on the result column */
    @Option(name="-c", aliases={"--result", "--header"}, metaVar="Result", usage="heading to put on result column")
    private String outColumn;

    /** model file name */
    @Option(name="--name", usage="model file name (default is model.ser in model directory)")
    private File modelName;

    /** output all confidences instead of output values */
    @Option(name="--regression", aliases={"-r"}, usage="output in regression mode rather than classification mode")
    private boolean confOutput;
    
    /** output file (if not STDOUT) */
    @Option(name="--output", aliases={"-o"}, usage="output file name (if not STDOUT)")
    private File outFile;

   /** model directory */
    @Argument(index=0, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;

    public static void makePredictions(File modelDir, boolean regression, File inFile, File outFile, 
    		List<String> metaList) throws IOException {
    	// Create the prediction processor and set up the parameters.
    	PredictionProcessor processor = new PredictionProcessor();
    	processor.modelDir = modelDir;
    	processor.inputFile = inFile;
    	processor.metaList = metaList;
    	processor.outColumn = "predicted";
    	processor.modelName = null;
    	processor.outFile = outFile;
    	processor.confOutput = regression;
    	// Open the output file.
    	processor.writer = new PrintStream(outFile);
    	// Initialize the prediction data.
    	processor.setupPredictionData();
    	// Produce the predictions.
    	processor.run();
    }
    
    

    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Set the defaults.
        this.help = false;
        this.inputFile = null;
        this.metaCols = "";
        this.outColumn = "predicted";
        this.modelName = null;
        this.outFile = null;
        // Parse the command line.
        CmdLineParser parser = new CmdLineParser(this);
        try {
            parser.parseArgument(args);
            if (this.help) {
                parser.printUsage(System.err);
            } else {
                // Verify the model directory and read the labels.
                if (! this.modelDir.isDirectory()) {
                    throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
                } else {
                    // Parse the metadata column list.
                    this.metaList = Arrays.asList(StringUtils.split(this.metaCols, ','));
                    // Set up the output print writer.
                    if (this.outFile == null) {
                    	log.info("Predictions will be written to the standard output.");
                    	this.writer = System.out;
                    } else {
                    	log.info("Predictions will be written to {}.", this.outFile);
                    	this.writer = new PrintStream(this.outFile);
                    }
                    // Read in the labels from the label file and set up the input stream.
                    setupPredictionData();
                    // Denote we're ready.
                    retVal = true;
                }
            }
        } catch (IOException e) {
            System.err.print(e.toString());
        } catch (CmdLineException e) {
            System.err.print("Invalid command-line options: " + e.toString());
        }
        return retVal;
    }

	/**
	 * Read the labels from the label file and set up the input.
	 * 
	 * @throws IOException
	 */
	protected void setupPredictionData() throws IOException {
		File labelFile = new File(this.modelDir, "labels.txt");
		if (! labelFile.exists()) {
		    throw new FileNotFoundException("Label file not found in " + this.modelDir + ".");
		} else {
		    this.labels = TabbedDataSetReader.readLabels(labelFile);
		    log.info("{} labels read from label file.", this.labels.size());
		    // Read in the model and the normalizer.
		    if (this.modelName == null)
		        this.modelName = new File(this.modelDir, "model.ser");
		    this.model = ModelSerializer.restoreMultiLayerNetwork(this.modelName, false);
		    DataNormalization normalizer = ModelSerializer.restoreNormalizerFromFile(this.modelName);
		    log.info("Model read from {}.", this.modelName);
		    // Determine the input type and get the appropriate reader.
		    File channelFile = new File(this.modelDir, "channels.tbl");
		    if (! channelFile.exists()) {
		        log.info("Normal input.");
		        // Normal situation.  Read scalar values.
		        this.reader = new TabbedDataSetReader(this.inputFile, metaList);
		    } else {
		        log.info("Channel input.");
		        // Here we have channel input.
		        Map<String, double[]> channelMap = ChannelDataSetReader.readChannelFile(channelFile);
		        this.reader = new ChannelDataSetReader(this.inputFile, metaList, channelMap);
		    }
		    this.reader.setNormalizer(normalizer);
		    // Write the output headers.  Note that if there is only one output column, we use the
		    // supplied name.
		    String metaHeader = StringUtils.join(this.metaList, '\t');
		    String confColumns;
		    if (! this.confOutput)
		        confColumns = this.outColumn + "\tconfidence";
		    else if (this.labels.size() == 1)
		        confColumns = this.outColumn;
		    else
		        confColumns = StringUtils.join(this.labels, '\t');
		    this.writer.format("%s\t%s%n", metaHeader, confColumns);
		}
	}

    @Override
    public void run() {
    	try {
	        // Loop through the data batches.
	        long start = System.currentTimeMillis();
	        int rows = 0;
	        for (DataSet batch : this.reader) {
	            // Get the features and the associated metadata.
	            INDArray features = batch.getFeatures();
	            List<String> metaData = batch.getExampleMetaData(String.class);
	            // Compute the predictions for this model.
	            INDArray output = model.output(features);
	            // Loop through the output and the metadata in parallel.
	            int i = 0;
	            for (String metaDatum : metaData) {
	                // We have the metadata for this row.  Find the output.
	                if (this.confOutput) {
	                    // Here we need to output the confidences column by column.
	                    // Start with the metadata.
	                    this.writer.print(metaDatum);
	                    // Loop through the labels.
	                    for (int j = 0; j < this.labels.size(); j++)
	                        this.writer.format("\t%12.8g", output.getDouble(i, j));
	                    // Terminate the line.
	                    this.writer.println();
	                } else {
	                    // Here we need to find the best label and its confidence.
	                    int n = this.labels.size();
	                    int jBest = 0;
	                    double vBest = output.getDouble(i, 0);
	                    for (int j = 1; j < n; j++) {
	                        double v = output.getDouble(i, j);
	                        if (v > vBest) {
	                            vBest = v;
	                            jBest = j;
	                        }
	                    }
	                    String prediction = this.labels.get(jBest);
	                    this.writer.format("%s\t%s\t%12.8g%n", metaDatum, prediction, vBest);
	                }
	                // Advance the row index and count.
	                i++;
	                rows++;
	            }
	        }
	        log.info("{} data rows processed in {} seconds.", rows, (System.currentTimeMillis() - start) / 1000);
    	} finally {
    		// Insure we close the output stream.
    		this.writer.close();
    	}
    }
}
