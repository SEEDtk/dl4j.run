/**
 *
 */
package org.theseed.reports;

import java.io.OutputStream;
import java.util.Arrays;
import java.util.stream.Collectors;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.theseed.dl4j.train.IPredictError;

/**
 * This is a base class for validation reports.  It contains a few methods common to all of them.
 * @author Bruce Parrello
 *
 */
public abstract class BaseValidationReport extends BaseReporter implements IValidationReport {

    // FIELDS
    /** logging facility */
    protected static Logger log = LoggerFactory.getLogger(ClassValidationReport.class);

    /**
     * Construct a validation reporter for a specified output stream.
     *
     * @param output	output stream to receive the report
     */
    public BaseValidationReport(OutputStream output) {
        super(output);
    }

    @Override
    public void finishReport(IPredictError errors) {
        log.info("Mean error from validation = {}.", errors.getError());
        log.info("**{}", Arrays.stream(errors.getTitles()).map(x -> String.format(" %14s", x)).collect(Collectors.joining()));
        log.info("**{}", Arrays.stream(errors.getStats()).mapToObj(x -> String.format(" %14.8f", x)).collect(Collectors.joining()));
    }

}
