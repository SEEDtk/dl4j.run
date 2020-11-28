/**
 *
 */
package org.theseed.dl4j.train;

import java.io.FileNotFoundException;
import java.io.IOException;

import org.apache.commons.text.TextStringBuilder;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.theseed.dl4j.TabbedDataSetReader;
import org.theseed.utils.ICommand;

/**
 * This class loads a pre-trained classification model and improves it by applying a new training set.  Most of the code is
 * inherited from the superclass LearningProcessor.
 *
 * @author Bruce Parrello
 *
 */
public class ImproveProcessor extends LearningProcessor implements ICommand {

    /** optimization preference */
    @Option(name = "--prefer", metaVar = "SCORE", usage = "model aspect to optimize during search")
    protected RunStats.OptimizationType preference;
    /** name or index of the label column */
    @Option(name = "-c", aliases = { "--col" }, metaVar = "0", usage = "input column containing class")
    protected String labelCol;

    @Override
    public boolean parseCommand(String[] args) {
        // Set the defaults.
        this.preference = RunStats.OptimizationType.ACCURACY;
        this.labelCol = "1";
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
                    this.setupTraining(labelCol);
                    TabbedDataSetReader myReader = this.openReader(this.trainingFile, this.labelCol);
                    this.initializeReader(myReader);
                    // Read in the testing set.
                    readTestingSet();
                }
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
            MultiLayerNetwork model = readModel();
            this.reader.setNormalizer(this.getNormalizer());
            // Now  we train the model.
            Trainer trainer = Trainer.create(this.method, this, log);
            RunStats runStats = RunStats.create(model, this.preference, trainer);
            this.trainModel(model, runStats, trainer);
            this.saveModel();
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
                this.clearRating();
            } else {
                this.accuracyReport(bestModel, parms, runStats);
            }
            // Add the summary.
            parms.appendln(bestModel.summary());
            // Add the parameter dump.
            parms.append(this.dumpModel(bestModel));
            // Output the result.
            String report = parms.toString();
            log.info(report);
            RunStats.writeTrialReport(this.getTrialFile(), this.comment, report);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }

}
