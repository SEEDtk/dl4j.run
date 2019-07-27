/**
 *
 */
package org.theseed.dl4j.predict;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

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
 * -b	size of each input batch; the default is 500; a larger batch size improves throughput, but a smaller
 * 		one reduces the memory footprint
 * -i	the name of the input file of predictions; the default is the standard input
 * -o	heading to put on the result column; the default is "predicted"
 *
 * --meta	a comma-delimited list of the metadata columns; these columns are ignored during training; the
 * 			default is none

 * @author Bruce Parrello
 *
 */
public class PredictionProcessor implements ICommand {

    // FIELDS
    /** list of metadata column labels */
    List<String> metaList;
    /** array of labels */
    List<String> labels;
    /** input dataset reader */
    TabbedDataSetReader reader;
    /** model to use for predictions */
    MultiLayerNetwork model;

    /** logging facility */
    private static Logger log = LoggerFactory.getLogger(PredictionProcessor.class);

    // COMMAND LINE

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    /** size of each input batch */
    @Option(name="-b", aliases={"--batchSize"}, metaVar="1000",
            usage="size of each input batch")
    private int batchSize;

    /** input prediction set */
    @Option(name="-i", aliases={"--input"}, metaVar="requests.tbl",
            usage="input prediction set file")
    private File inputFile;

    /** comma-delimited list of metadata column names */
    @Option(name="--meta", metaVar="name,date", usage="comma-delimited list of metadata columns")
    private String metaCols;

    /** heading to put on the result column */
    @Option(name="-o", aliases={"--result", "--header"}, metaVar="Result", usage="heading to put on result column")
    private String outColumn;

    /** model directory */
    @Argument(index=0, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;



    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Set the defaults.
        this.help = false;
        this.batchSize = 500;
        this.inputFile = null;
        this.metaCols = "";
        this.outColumn = "predicted";
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
                    // Read in the labels from the label file.
                    File labelFile = new File(this.modelDir, "labels.txt");
                    if (! labelFile.exists()) {
                        throw new FileNotFoundException("Label file not found in " + this.modelDir + ".");
                    } else {
                        this.labels = TabbedDataSetReader.readLabels(labelFile);
                        log.info("{} labels read from label file.", this.labels.size());
                        // Parse the metadata column list.
                        this.metaList = Arrays.asList(StringUtils.split(this.metaCols, ','));
                        // Read in the model and the normalizer.
                        File modelFile = new File(modelDir, "model.ser");
                        this.model = ModelSerializer.restoreMultiLayerNetwork(modelFile, false);
                        DataNormalization normalizer = ModelSerializer.restoreNormalizerFromFile(modelFile);
                        // Open the reader.
                        // Determine the input type and get the appropriate reader.
                        File channelFile = new File(this.modelDir, "channels.tbl");
                        if (! channelFile.exists()) {
                            log.info("Normal input.");
                            // Normal situation.  Read scalar values.
                            this.reader = new TabbedDataSetReader(this.inputFile, metaList);
                        } else {
                            log.info("Channel input.");
                            // Here we have channel input.
                            HashMap<String, double[]> channelMap = ChannelDataSetReader.readChannelFile(channelFile);
                            this.reader = new ChannelDataSetReader(this.inputFile, metaList, channelMap);
                        }
                        this.reader.setNormalizer(normalizer);
                        // Write the output headers.
                        String metaHeader = StringUtils.join(this.metaList, '\t');
                        System.out.format("%s\t%s\tconfidence%n", metaHeader, this.outColumn);
                        // Denote we're ready.
                        retVal = true;
                    }
                }
            }
        } catch (IOException e) {
            System.err.print(e.getMessage());
        } catch (CmdLineException e) {
            System.err.print("Invalid command-line options: " + e.getMessage());
        }
        return retVal;
    }

    @Override
    public void run() {
        // Loop through the data batches.
        for (DataSet batch : this.reader) {
            // Get the features and the associated metadata.
            INDArray features = batch.getFeatures();
            List<String> metaData = batch.getExampleMetaData(String.class);
            // Compute the predictions for this model.
            INDArray output = model.output(features);
            // Loop through the output and the metadata in parallel.
            int i = 0;
            for (String metaDatum : metaData) {
                // Find the best output for this row.
                int jBest = 0;
                double vBest = output.getDouble(i, 0);
                for (int j = 1; j < this.labels.size(); j++) {
                    double v = output.getDouble(i, j);
                    if (v > vBest) {
                        vBest = v;
                        jBest = j;
                    }
                }
                String prediction = this.labels.get(jBest);
                System.out.format("%s\t%s\t%8.4g%n", metaDatum, prediction, vBest);
            }
        }
    }
}
