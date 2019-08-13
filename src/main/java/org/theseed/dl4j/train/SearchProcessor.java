/**
 *
 */
package org.theseed.dl4j.train;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.List;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.theseed.dl4j.App;
import org.theseed.utils.ICommand;
import org.theseed.utils.Parms;

/**
 * In search mode, we accept a parameter file that specifies multiple values for one or more
 * parameters.  We run the training processor on each value combination.  This should be done
 * with a small number of iterations to keep performance rational.
 *
 * There are two positional parameters-- the name of the parameter file and the name of the
 * model directory.  The parameters in the parameter file are defined by the TrainingProcessor
 * class, however, no value should be specified for the "--comment" option, and the model
 * directory name will come from the command line, not the parameter file.
 *
 * @author Bruce Parrello
 *
 */
public class SearchProcessor implements ICommand {

    // FIELDS
    /** iterator that runs through the parameter combinations */
    private Parms parmIterator;

    // COMMAND LINE

    /** help option */
    @Option(name="-h", aliases={"--help"}, help=true)
    private boolean help;

    @Argument(index=0, metaVar="parms.prm", usage="parameter file with tab-separated alternatives", required=true)
    private File parmFile;

    /** model directory */
    @Argument(index=1, metaVar="modelDir", usage="model directory", required=true)
    private File modelDir;

    @Override
    public boolean parseCommand(String[] args) {
        boolean retVal = false;
        // Parse the command line.
        CmdLineParser parser = new CmdLineParser(this);
        try {
            this.help = false;
            parser.parseArgument(args);
            if (this.help) {
                parser.printUsage(System.err);
            } else {
                // Verify the model directory and read the parm file.
                if (! this.modelDir.isDirectory()) {
                    throw new FileNotFoundException("Model directory " + this.modelDir + " not found or invalid.");
                } else {
                    this.parmIterator = new Parms(this.parmFile);
                    // Denote we have been successful.
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
        // We loop through the parameter combinations, calling the training processor.
        TrainingProcessor processor = new TrainingProcessor();
        // These variables track our progress and success.
        int iteration = 1;
        double bestAccuracy = 0.0;
        int bestIteration = 0;
        // This is a buffer to hold the parameters.
        String[] parmBuffer = new String[this.parmIterator.size() + 3];
        while (this.parmIterator.hasNext()) {
            // Create the parameter array.
            List<String> theseParms = this.parmIterator.next();
            // Add the comment and the model directory.
            theseParms.add("--comment");
            theseParms.add(String.format("Iteration %d", iteration));
            theseParms.add(this.modelDir.getPath());
            // Run the experiment.
            String[] actualParms = theseParms.toArray(parmBuffer);
            boolean success = App.execute(processor, actualParms);
            if (success) {
                // Compare the accuracy.
                double newAccuracy = processor.getAccuracy();
                if (newAccuracy > bestAccuracy) {
                    bestIteration = iteration;
                    bestAccuracy = newAccuracy;
                }
            }
            // Count the iteration.
            iteration++;
        }
        TrainingProcessor.log.info("** Best iteration was {} with accuracy {}.", bestIteration, bestAccuracy);
    }

}
