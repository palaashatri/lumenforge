package atri.palaash.jforge.core.tokenizer;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.file.Path;
import java.text.Normalizer;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class T5Tokenizer {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();

    private final Map<String, Integer> pieceToId;
    private final float[] scores;
    private final int padId;
    private final int eosId;
    private final int unkId;
    private final int maxPieceLen;

    private T5Tokenizer(Map<String, Integer> pieceToId, float[] scores,
                        int padId, int eosId, int unkId, int maxPieceLen) {
        this.pieceToId = pieceToId;
        this.scores = scores;
        this.padId = padId;
        this.eosId = eosId;
        this.unkId = unkId;
        this.maxPieceLen = maxPieceLen;
    }

    @SuppressWarnings("unchecked")
    public static T5Tokenizer load(Path tokenizerJsonPath) throws Exception {
        Map<String, Object> root = OBJECT_MAPPER.readValue(
                tokenizerJsonPath.toFile(), new TypeReference<>() {
                });
        Map<String, Object> modelSection = (Map<String, Object>) root.get("model");
        if (modelSection == null) {
            throw new IllegalArgumentException("tokenizer.json has no 'model' section");
        }
        List<List<Object>> vocab = (List<List<Object>>) modelSection.get("vocab");
        if (vocab == null || vocab.isEmpty()) {
            throw new IllegalArgumentException("tokenizer.json model has no 'vocab'");
        }

        int unkIdFromModel = modelSection.get("unk_id") instanceof Number n ? n.intValue() : 2;

        Map<String, Integer> pieceToId = new HashMap<>(vocab.size());
        float[] scoreArr = new float[vocab.size()];
        int maxLen = 1;
        for (int i = 0; i < vocab.size(); i++) {
            List<Object> entry = vocab.get(i);
            String piece = (String) entry.get(0);
            float score = ((Number) entry.get(1)).floatValue();
            pieceToId.put(piece, i);
            scoreArr[i] = score;
            maxLen = Math.max(maxLen, piece.length());
        }

        // T5 usually uses 0=pad, 1=eos, 2=unk
        return new T5Tokenizer(pieceToId, scoreArr, 0, 1, unkIdFromModel, maxLen);
    }

    public long[] encode(String text, int maxLength) {
        if (text == null) text = "";

        // 1. Normalize (NFKC)
        String normalized = Normalizer.normalize(text, Normalizer.Form.NFKC);

        // 2. Pre-tokenize: Metaspace (replace space with   (U+2581))
        // And prepend a space if it doesn't start with one (T5 convention)
        String t5Text = " " + normalized.replace(" ", "\u2581");

        // 3. Viterbi segmentation (Unigram)
        int n = t5Text.length();
        float[] bestScore = new float[n + 1];
        int[] bestPos = new int[n + 1];
        int[] bestId = new int[n + 1];

        Arrays.fill(bestScore, Float.NEGATIVE_INFINITY);
        bestScore[0] = 0;

        for (int i = 0; i < n; i++) {
            if (bestScore[i] == Float.NEGATIVE_INFINITY) continue;

            for (int len = 1; len <= maxPieceLen && i + len <= n; len++) {
                String piece = t5Text.substring(i, i + len);
                Integer id = pieceToId.get(piece);
                if (id != null) {
                    float score = bestScore[i] + scores[id];
                    if (score > bestScore[i + len]) {
                        bestScore[i + len] = score;
                        bestPos[i + len] = i;
                        bestId[i + len] = id;
                    }
                }
            }
            // If no piece found, skip 1 char as unknown
            if (bestScore[i + 1] == Float.NEGATIVE_INFINITY) {
                float score = bestScore[i] + scores[unkId] - 10f; // penalty
                bestScore[i + 1] = score;
                bestPos[i + 1] = i;
                bestId[i + 1] = unkId;
            }
        }

        // Backtrack
        java.util.LinkedList<Integer> ids = new java.util.LinkedList<>();
        int curr = n;
        while (curr > 0) {
            ids.addFirst(bestId[curr]);
            curr = bestPos[curr];
        }

        // Add EOS
        ids.add(eosId);

        // Truncate/Pad
        if (ids.size() > maxLength) {
            ids = new java.util.LinkedList<>(ids.subList(0, maxLength));
            ids.set(maxLength - 1, eosId);
        }

        long[] result = new long[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            result[i] = ids.get(i);
        }
        return result;
    }
}
