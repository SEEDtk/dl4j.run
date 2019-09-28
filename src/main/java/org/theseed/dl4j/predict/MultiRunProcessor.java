/**
 *
 */
package org.theseed.dl4j.predict;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

import org.apache.commons.lang3.StringUtils;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.ChannelDataSetReader;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.io.TabbedLineReader;
import org.theseed.utils.ICommand;

/**
 * This command runs multiple models against parallel input files.  In each file a single
 * classification column is chosen as output and an input meta-data column is chosen as
 * the key.  The scores are collated by key and output in a single row.
 *
 * Each line of the input file describes a single model run and contains the following
 * fields.
 *
 * <ol>
 * <li>the filename of the model to run</li>
 * <li>the filename of the input</li>
 * <li>the names of the metadata columns, comma-delimited</li>
 * <li>the title for the output column</li>
 * </ol>
 *
 * The positional parameter is the directory containing all the models.  The input and model file names
 * will be calculated relative to that directory.
 *
 * Note that all the models must use the same channel file and the same labels.
 *
 * The following command-line options are supported.
 *
 * -i	the name of the input file of runs to make; the default is the standard input
 * -r	the label containing the desired result value; the default is "yes"
 * -k	the name of the key column in the prediction input files
 * -m	minimum acceptable confidence
 *
 * --nohead		if specified, it is assumed the input file has no header line
 *
 * @author Bruce Parrello
 *
 */
public class MultiRunProcessor implements ICommand {

    // FIELDS
    /** array of labels */
    private List<String> labels;
    /** channel translation map */
    Map<String, double[]> channelMap;
    /** index of key in label list */
    private int classIdx;
    /** input file reader */
    private TabbedLineReader inStream;

    // CONSTANTS

    /** logging facility */
    private static Logger log = LoggerFactory.getLogger(MultiRunProcessor.class);


    // COMMAND-LINE OPTIONS

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    /** input prediction set */
    @Option(name="-i", aliases={"--input"}, metaVar="requests.tbl",
            usage="input prediction set file (default STDIN)")
    private File inputFile;

    /** desired class label name */
    @Option(name="-r", aliases={"--result"}, metaVar="start", usage="relevant result class")
    private String keyClass;

    /** prediction input file key column */
    @Option(name="-k", aliases={"--keyCol"}, metaVar="id", usage="prediction input file key column")
    private String keyCol;

    /** no-headers flag */
    @Option(name="--nohead", usage="indicates input file has no headers")
    private boolean noHeaders;

    /** minimum acceptable confidence */
    @Option(name="-m", aliases={"--min"}, metaVar="0.5", usage="minimum acceptable confidence")
    private double minConf;

   /** model directory */
    @Argument(index=0, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;

    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Set the defaults.
        this.help = false;
        this.inputFile = null;
        this.keyClass = "yes";
        this.keyCol = "1";
        this.minConf = 0.0;
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
                        // Determine the input type and get the channel map.
                        File channelFile = new File(this.modelDir, "channels.tbl");
                        if (! channelFile.exists()) {
                            log.info("Normal input.");
                            // Normal situation.  Read scalar values.
                            this.channelMap = null;
                        } else {
                            log.info("Channel input.");
                            // Here we have channel input.
                            this.channelMap = ChannelDataSetReader.readChannelFile(channelFile);
                        }
                        // Find the index of the main class column in the label list.
                        this.classIdx = this.labels.indexOf(this.keyClass);
                        if (this.classIdx < 0)
                            throw new IllegalArgumentException("Label \"" + this.keyClass +
                                    "\" not found in class label list.");
                        // Finally, open the input file.
                        if (this.inputFile == null) {
                            if (this.noHeaders) {
                                this.inStream = new TabbedLineReader(System.in, ControlLine.colCount());
                            } else {
                                this.inStream = new TabbedLineReader(System.in);
                            }
                        } else {
                            if (this.noHeaders) {
                                this.inStream = new TabbedLineReader(this.inputFile, ControlLine.colCount());
                            } else {
                                this.inStream = new TabbedLineReader(this.inputFile);
                            }
                        }
                        // We made it this far, we can proceed.
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
        try {
            log.info("Reading control file.");
            // This will contain the output headings.
            ArrayList<String> headings = new ArrayList<String>();
            headings.add("id");
            headings.add("best");
            // This will contain all the controllers.
            ArrayList<ControlLine> controllers = new ArrayList<ControlLine>();
            for (TabbedLineReader.Line line : this.inStream) {
                ControlLine controller = new ControlLine(line, this.modelDir, this.keyCol);
                controllers.add(controller);
                headings.add(controller.getOutputHeading());
            }
            this.inStream.close();
            log.info("{} models will be processed.", controllers.size());
            // This will hold a list of column values for each input key, sorted so
            // we can output the keys in order.
            Map<String, double[]> outputMap = new TreeMap<String, double[]>();
            // Loop through the controllers, applying the models.
            for (int col = 0; col < controllers.size(); col++) {
                ControlLine controller = controllers.get(col);
                log.info("Processing model for {}.", controller.getOutputHeading());
                controller.init(channelMap);
                TabbedDataSetReader reader = controller.getReader();
                MultiLayerNetwork model = controller.getModel();
                // Process all the data in the input file.
                for (DataSet batch : reader) {
                    // Get the input features and the metadata for this batch.
                    INDArray features = batch.getFeatures();
                    List<String> metaData = batch.getExampleMetaData(String.class);
                    // Compute the predictions.
                    INDArray output = model.output(features);
                    // Loop through the output and the metadata in parallel.
                    int row = 0;
                    for (String metaDatum : metaData) {
                        // Get the desired output for this row.
                        double confidence = output.getDouble(row, this.classIdx);
                        // Compute the key.
                        String key = StringUtils.substringBefore(metaDatum, "\t");
                        // Get the array entry for this key and store the result.
                        double[] results = outputMap.get(key);
                        if (results == null) {
                            results = new double[controllers.size()];
                            Arrays.fill(results, 0.0);
                            outputMap.put(key, results);
                        }
                        results[col] = confidence;
                        // Advance the row index.
                        row++;
                    }
                }
                // Release the memory for the model and the reader.
                controller.close();
            }
            // Now all the data has been processed, so we want to write it out.
            log.info("Writing output.");
            System.out.println(StringUtils.join(headings, "\t"));
            for (Map.Entry<String, double[]> keyDatum : outputMap.entrySet()) {
                // Get the best result.
                double[] results = keyDatum.getValue();
                double bestVal = this.minConf;
                String best = "";
                for (int i = 0; i < results.length; i++)
                    if (bestVal < results[i]) {
                        best = controllers.get(i).getOutputHeading();
                        bestVal = results[i];
                    }
                System.out.format("%s\t%s", keyDatum.getKey(), best);
                for (double result : results) {
                    System.out.format("\t%12.8f", result);
                }
                System.out.println();
            }
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }

}
