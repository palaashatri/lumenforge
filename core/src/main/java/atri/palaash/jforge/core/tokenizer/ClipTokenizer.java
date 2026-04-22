package atri.palaash.jforge.core.tokenizer;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class ClipTokenizer {
    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final Pattern TOKEN_PATTERN = Pattern.compile("'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+");

    private final Map<String, Integer> vocab;
    private final Map<String, Integer> merges;
    private final Map<Integer, String> byteEncoder;
    private final Map<String, String> cache = new java.util.LinkedHashMap<>(256, 0.75f, true) {
        @Override
        protected boolean removeEldestEntry(java.util.Map.Entry<String, String> eldest) {
            return size() > 10_000;
        }
    };
    private final int bos;
    private final int eos;

    private ClipTokenizer(Map<String, Integer> vocab, Map<String, Integer> merges) {
        this.vocab = vocab;
        this.merges = merges;
        this.byteEncoder = bytesToUnicode();
        this.bos = vocab.getOrDefault("<|startoftext|>", 49406);
        this.eos = vocab.getOrDefault("<|endoftext|>", 49407);
    }

    public static ClipTokenizer load(Path vocabPath, Path mergesPath) throws Exception {
        Map<String, Integer> vocab = OBJECT_MAPPER.readValue(vocabPath.toFile(), new TypeReference<>() {
        });
        List<String> mergeLines = java.nio.file.Files.readAllLines(mergesPath, StandardCharsets.UTF_8);
        Map<String, Integer> ranks = new HashMap<>();
        int rank = 0;
        for (String line : mergeLines) {
            String trimmed = line.trim();
            if (trimmed.isBlank() || trimmed.startsWith("#")) {
                continue;
            }
            ranks.put(trimmed, rank++);
        }
        return new ClipTokenizer(vocab, ranks);
    }

    public long[] encode(String text, int maxLength) {
        List<Integer> ids = new ArrayList<>();
        ids.add(bos);
        String normalized = text == null ? "" : text.toLowerCase();
        Matcher matcher = TOKEN_PATTERN.matcher(normalized);
        while (matcher.find()) {
            String token = matcher.group();
            StringBuilder encoded = new StringBuilder();
            byte[] bytes = token.getBytes(StandardCharsets.UTF_8);
            for (byte value : bytes) {
                encoded.append(byteEncoder.get(value & 0xFF));
            }
            String bpeToken = bpe(encoded.toString());
            for (String piece : bpeToken.split(" ")) {
                Integer id = vocab.get(piece);
                if (id != null) {
                    ids.add(id);
                }
            }
        }
        while (ids.size() < maxLength) {
            ids.add(eos);
        }
        if (ids.size() > maxLength) {
            ids = ids.subList(0, maxLength);
            ids.set(maxLength - 1, eos);
        }
        long[] result = new long[ids.size()];
        for (int i = 0; i < ids.size(); i++) {
            result[i] = ids.get(i);
        }
        return result;
    }

    private String bpe(String token) {
        if (cache.containsKey(token)) {
            return cache.get(token);
        }
        List<String> words = new ArrayList<>();
        for (int i = 0; i < token.length(); i++) {
            words.add(String.valueOf(token.charAt(i)));
        }
        while (words.size() > 1) {
            String bestPair = null;
            int minRank = Integer.MAX_VALUE;
            for (int i = 0; i < words.size() - 1; i++) {
                String pair = words.get(i) + " " + words.get(i + 1);
                Integer rank = merges.get(pair);
                if (rank != null && rank < minRank) {
                    minRank = rank;
                    bestPair = pair;
                }
            }
            if (bestPair == null) {
                break;
            }
            String[] parts = bestPair.split(" ");
            List<String> nextWords = new ArrayList<>();
            for (int i = 0; i < words.size(); i++) {
                if (i < words.size() - 1 && words.get(i).equals(parts[0]) && words.get(i + 1).equals(parts[1])) {
                    nextWords.add(parts[0] + parts[1]);
                    i++;
                } else {
                    nextWords.add(words.get(i));
                }
            }
            words = nextWords;
        }
        String result = String.join(" ", words);
        cache.put(token, result);
        return result;
    }

    private static Map<Integer, String> bytesToUnicode() {
        Map<Integer, String> map = new HashMap<>();
        for (int b = (int) '!' ; b <= (int) '~' ; b++) map.put(b, String.valueOf((char) b));
        for (int b = (int) '\u00a1' ; b <= (int) '\u00ac' ; b++) map.put(b, String.valueOf((char) b));
        for (int b = (int) '\u00ae' ; b <= (int) '\u00ff' ; b++) map.put(b, String.valueOf((char) b));
        int n = 0;
        for (int b = 0; b < 256; b++) {
            if (!map.containsKey(b)) {
                map.put(b, String.valueOf((char) (256 + n++)));
            }
        }
        return map;
    }
}
