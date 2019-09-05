/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.commons.text.TextStringBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.theseed.utils.ICommand;

/**
 * This class loads a pre-trained model and improves it by applying a new training set.  Most of the code is
 * inherited from the superclass LearningProcessor.
 *
 * @author Bruce Parrello
 *
 */
public class ImproveProcessor extends LearningProcessor implements ICommand {

    @Override
    public boolean parseCommand(String[] args) {
        // Set the defaults.
        this.setDefaults();
        // This will be the return value.
        boolean retVal = false;
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
                    this.setupTraining();
                }
                // TODO more parameter processing
                // We made it this far, we can run the application.
                retVal = true;
            }
        } catch (CmdLineException e) {
            System.err.println(e.getMessage());
            // For parameter errors, we display the command usage.
            parser.printUsage(System.err);
        } catch (IOException e) {
            System.err.println(e.getMessage());
        }
        return retVal;
    }

    @Override
    public void run() {
        try {
            // Read in the model and the normalizer.
            if (this.modelName == null)
                this.modelName = new File(this.modelDir, "model.ser");
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(this.modelName, false);
            DataNormalization normalizer = ModelSerializer.restoreNormalizerFromFile(this.modelName);
            // Now  we train the model.
            RunStats runStats = trainModel(normalizer, model);
            // Display the configuration.
            MultiLayerNetwork bestModel = runStats.getBestModel();
            TextStringBuilder parms = new TextStringBuilder();
            parms.appendNewLine();
            parms.appendln(
                            "=========================== Parameters ===========================%n" +
                            "     iterations  = %12d, batch size    = %12d%n" +
                            "     test size   = %12d, seed number   = %12d%n" +
                            "     --------------------------------------------------------%n" +
                            "     Training set strategy is %s.%n" +
                            "     %s minutes to run %d %s (best was %d), with %d score bounces.%n" +
                            "     %d models saved.",
                           this.iterations, this.batchSize, this.testSize, this.seed,
                           this.method.toString(), runStats.getDuration(), runStats.getEventCount(),
                           runStats.getEventsName(), runStats.getBestEvent(), runStats.getBounceCount(),
                           runStats.getSaveCount());
            if (this.isChannelMode())
                parms.appendln("     Input uses channel vectors.");
            if (runStats.getSaveCount() == 0) {
                parms.appendln("%nMODEL FAILED DUE TO OVERFLOW OR UNDERFLOW.");
                this.clearAccuracy();
            } else {
                this.accuracyReport(bestModel, parms);
            }
            // Add the summary.
            parms.appendln(bestModel.summary());
            // Add the parameter dump.
            parms.append(this.dumpModel(bestModel));
            // Output the result.
            String report = parms.toString();
            log.info(report);
            RunStats.writeTrialReport(this.modelDir, this.comment, report);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }

}