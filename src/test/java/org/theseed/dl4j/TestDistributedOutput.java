/**
 *
 */
package org.theseed.dl4j;

import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.*;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import org.apache.commons.lang3.StringUtils;
import org.junit.Test;
import org.theseed.counters.QualityCountMap;
import org.theseed.dl4j.train.ClassTrainingProcessor;
import org.theseed.dl4j.train.RegressionTrainingProcessor;
import org.theseed.io.LineReader;

/**
 * @author Bruce Parrello
 *
 */
public class TestDistributedOutput {

    @Test
    public void testDiscrete() throws IOException {
        // Run this test 10 times because of the randomness.
        for (int run = 0; run < 10; run++) {
            String[] headers;
            String headLine;
            try (LineReader headReader = new LineReader(new File("src/test/data", "headers.tbl"))) {
                String line = headReader.next();
                headers = StringUtils.split(line, '\t');
                headLine = line;
            }
            File outFile = new File("src/test/data", "data.ser");
            Map<String, String> inIDs = new HashMap<String, String>(350);
            try (LineReader dataReader = new LineReader(new File("src/test/data", "raw.data"))) {
                DistributedOutputStream outStream = DistributedOutputStream.create(outFile, new ClassTrainingProcessor(), "loc", headers);
                for (String line : dataReader) {
                    String[] items = StringUtils.split(line, ',');
                    inIDs.put(items[0], StringUtils.replaceChars(line, ',', '\t'));
                    outStream.write(items);
                }
                outStream.close();
                assertThat(outStream.getOutputCount(), equalTo(inIDs.size()));
            }
            // These maps will count the occurrences of each label.
            QualityCountMap<String> counters = new QualityCountMap<String>();
            int halfway = inIDs.size() / 2;
            try (LineReader dataReader = new LineReader(outFile)) {
                int i = 0;
                Iterator<String> iter = dataReader.iterator();
                String header = iter.next();
                assertThat(header, equalTo(headLine));
                while (iter.hasNext()) {
                    String line = iter.next();
                    String key = StringUtils.substringBefore(line, "\t");
                    assertThat(inIDs.get(key), equalTo(line));
                    String loc = StringUtils.substringAfterLast(line, "\t");
                    if (i > halfway)
                        counters.setGood(loc);
                    else
                        counters.setBad(loc);
                    i++;
                }
                // Insure we output everything.
                assertThat(i, equalTo(inIDs.size()));
                // Insure each half has nearly the same amount of every key.
                for (String loc : counters.allKeys())
                    assertThat("Key " + loc, Math.abs(counters.good(loc) - counters.bad(loc)), lessThan(2));
            }
        }
    }

    @Test
    public void testContinuous() throws IOException {
        // Run this test 10 times because of the randomness.
        for (int run = 0; run < 10; run++) {
            String[] headers;
            String headLine;
            File outFile = new File("src/test/data", "data.ser");
            Map<String, String> inIDs = new HashMap<String, String>(350);
            try (LineReader dataReader = new LineReader(new File("src/test/data", "continuous.tbl"))) {
                String line = dataReader.next();
                headers = StringUtils.split(line, '\t');
                headLine = line;
                // Now read the input and create the output.
                DistributedOutputStream outStream = DistributedOutputStream.create(outFile, new RegressionTrainingProcessor(), "production", headers);
                while (dataReader.hasNext()) {
                    line = dataReader.next();
                    String[] items = StringUtils.split(line, '\t');
                    inIDs.put(items[0], line);
                    outStream.write(items);
                }
                outStream.close();
                assertThat(outStream.getOutputCount(), equalTo(inIDs.size()));
            }
            // Count the two extremes.
            int[] lowCount = new int[] { 0, 0 };
            int[] highCount = new int[] { 0, 0 };
            int halfway = (inIDs.size() + 1) / 2;
            try (LineReader dataReader = new LineReader(outFile)) {
                int i = 0;
                Iterator<String> iter = dataReader.iterator();
                String header = iter.next();
                assertThat(header, equalTo(headLine));
                while (iter.hasNext()) {
                    String line = iter.next();
                    String key = StringUtils.substringBefore(line, "\t");
                    assertThat(inIDs.get(key), equalTo(line));
                    double prod = Double.valueOf(StringUtils.substringAfterLast(line, "\t"));
                    int half = i / halfway;
                    if (prod == 0.0)
                        lowCount[half]++;
                    if (prod >= 0.765)
                        highCount[half]++;
                    i++;
                }
                // Insure we output everything.
                assertThat(i, equalTo(inIDs.size()));
                // Insure each half has a good representation of the values.
                assertThat(Math.abs(lowCount[0] - lowCount[1]), lessThan(2));
                assertThat(Math.abs(highCount[0] - highCount[1]), lessThan(2));
            }
        }

    }
}
