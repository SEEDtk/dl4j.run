/**
 *
 */
package org.theseed.dl4j.train;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.SortedSet;

import org.junit.jupiter.api.Test;
import org.theseed.counters.Rating;
import org.theseed.utils.ParseFailureException;

/**
 * @author Bruce Parrello
 *
 */
public class TestAnova {

    @Test
    public void test() throws IOException, ParseFailureException {
        RandomForestTrainProcessor processor = new RandomForestTrainProcessor();
        File modelDir = new File("src/test/data", "RnaSeqWide");
        SortedSet<Rating<String>> ratings = processor.computeAnovaFValues(modelDir);
        File anovaFile = new File(modelDir, "anova.txt");
        try (PrintWriter anovaStream = new PrintWriter(anovaFile)) {
            anovaStream.println("col_name\tf_measure");
            for (Rating<String> rating : ratings)
                anovaStream.format("%s\t%14.6f%n", rating.getKey(), rating.getRating());
        }
        assertThat(ratings.size(), greaterThan(2000));
    }

}
