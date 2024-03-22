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
import java.util.SortedSet;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.Option;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.basic.BaseProcessor;
import org.theseed.basic.ParseFailureException;
import org.theseed.dl4j.train.RandomForestTrainProcessor;
import org.theseed.stats.Rating;

/**
 * This is a simple command that outputs an ANOVA F-measure report for a random forest trainer.
 *
 * The positional parameter is the name of the model directory.  The following command-line parameters
 * are supported.
 *
 * -h	display command-line usage
 * -v	display more detailed log messages
 * -o	name of output file; the default is to write to STDOUT
 *
 * @author Bruce Parrello
 *
 */
public class AnovaProcessor extends BaseProcessor {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(AnovaProcessor.class);
    /** randon forest trainer */
    private RandomForestTrainProcessor processor;
    /** output stream */
    private OutputStream outStream;

    // COMMAND-LINE OPTIONS

    /** output file (if not STDOUT) */
    @Option(name = "-o", aliases = { "--output" }, metaVar = "report.txt", usage = "if specified, file to contain output")
    private File outFile;

    /** model directory */
    @Argument(index = 0, metaVar = "modelDir", usage = "random forest model directory", required = true)
    private File modelDir;

    @Override
    protected void setDefaults() {
        this.outFile = null;
    }

    @Override
    protected boolean validateParms() throws IOException, ParseFailureException {
        // Insure the model directory is valid.
        if (! this.modelDir.isDirectory())
            throw new FileNotFoundException("Model directory " + this.modelDir + " is not found or invalid.");
        File parmFile = new File(modelDir, "parms.prm");
        if (! parmFile.canRead())
            throw new FileNotFoundException("Parameter file " + parmFile + " is not found or unreadable.");
        // Connect the output stream.
        if (this.outFile == null) {
            log.info("Output will be to the standard output.");
            this.outStream = System.out;
        } else {
            log.info("Output will be to {}.", this.outFile);
            this.outStream = new FileOutputStream(this.outFile);
        }
        return true;
    }

    @Override
    protected void runCommand() throws Exception {
        this.processor = new RandomForestTrainProcessor();
        log.info("Computing F-measures.");
        SortedSet<Rating<String>> ratings = processor.computeAnovaFValues(modelDir);
        log.info("Writing output.");
        try (PrintWriter anovaStream = new PrintWriter(this.outStream)) {
            anovaStream.println("col_name\tf_measure");
            for (Rating<String> rating : ratings)
                anovaStream.format("%s\t%14.6f%n", rating.getKey(), rating.getRating());
        }
    }

}
