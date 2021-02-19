/**
 *
 */
package org.theseed.dl4j.predict;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.train.ITrainingProcessor;
import org.theseed.dl4j.train.ModelType;
import org.theseed.reports.IValidationReport;
import org.theseed.utils.ICommand;
import org.theseed.utils.Parms;

/**
 * This command applies an existing model to a formatted training/testing set.  The output shows a comparison between the predicted and actual
 * results.
 *
 * The positional parameter is the name of the model directory.
 *
 * The command-line options are as follows.
 *
 * -h	display command-line usage
 * -t	type of model (REGRESSION or CLASS, default CLASS)
 * -i	file containing the training/testing set (default is to use the parameter file value)
 *
 * --parms name of the parameter file (the default is "parms.prm" in the model directory)
 *
 * @author Bruce Parrello
 *
 */
public class ValidateProcessor implements ICommand {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ValidateProcessor.class);
    /** parameter object */
    private Parms parms;
    /** actual input file */
    private File trainingFile;

    // COMMAND-LINE OPTIONS

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    /** type of model */
    @Option(name="--type", aliases={"-t"}, usage="type of model")
    private ModelType modelType;

    /** input training file (if not the one in the parm file */
    @Option(name = "-i", aliases = { "--input" }, metaVar = "training.tbl", usage = "input training file (overrides parms.prm if specified)")
    private File inFile;

    /** parameter file */
    @Option(name = "--parms", metaVar="parms.prm", usage="parameter file (if not the default)")
    private File parmFile;

    /** model directory */
    @Argument(index=0, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;

    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Parse the command line.
        CmdLineParser parser = new CmdLineParser(this);
        try {
            this.help = false;
            this.inFile = null;
            this.modelType = ModelType.CLASS;
            this.parmFile = null;
            parser.parseArgument(args);
            if (this.help) {
                parser.printUsage(System.err);
            } else {
                // Verify the model directory.
                if (! this.modelDir.isDirectory())
                    throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
                // Read the parameter file.
                if (this.parmFile == null)
                    this.parmFile = new File(this.modelDir, "parms.prm");
                log.info("Reading parameters from {}.", this.parmFile);
                if (! this.parmFile.canRead())
                    throw new FileNotFoundException("Parameter file " + this.parmFile + " not found or unreadable.");
                else {
                    this.parms = new Parms(this.parmFile);
                    if (this.inFile != null) {
                        if (! this.inFile.canRead())
                            throw new FileNotFoundException("Input file " + this.inFile + " is not found or unreadable.");
                        this.trainingFile = this.inFile;
                    } else {
                        // Extract the training file from the parameters.
                        String trainingName = this.parms.getValue("--training");
                        if (trainingName.isEmpty())
                            this.trainingFile = new File(this.modelDir, "training.tbl");
                        else
                            this.trainingFile = new File(trainingName);
                    }
                    retVal = true;
                }
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
            log.info("Input will be read from {}.", this.trainingFile);
            // Create the training processor.
            ITrainingProcessor processor = ModelType.create(this.modelType);
            // Set up the parameters.
            processor.setupParameters(this.parms, this.modelDir);
            // Verify channel mode.
            processor.checkChannelMode();
            // Create the reporter.
            IValidationReport reporter = processor.getValidationReporter(System.out);
            // Perform the prediction test.  The output goes to the reporter object.
            processor.runPredictions(reporter, this.trainingFile);
        } catch (Exception e) {
            e.printStackTrace(System.err);
        }
    }

}
