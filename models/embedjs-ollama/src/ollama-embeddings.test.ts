import assert from 'node:assert/strict';
import test from 'node:test';

import { OllamaEmbeddings } from './ollama-embeddings.js';

function createJsonResponse(payload: unknown, init?: ResponseInit): Response {
    return new Response(JSON.stringify(payload), {
        status: 200,
        headers: {
            'Content-Type': 'application/json',
        },
        ...init,
    });
}

function getRequestUrl(input: Parameters<typeof fetch>[0]): string {
    if (typeof input === 'string') {
        return input;
    }

    if (input instanceof URL) {
        return input.toString();
    }

    return input.url;
}

function stubFetch(responses: Response[]) {
    const calls: Array<{ url: string; init: RequestInit | undefined }> = [];
    const originalFetch = globalThis.fetch;

    globalThis.fetch = (async (input, init) => {
        calls.push({
            url: getRequestUrl(input),
            init,
        });

        const response = responses.shift();
        if (!response) {
            throw new Error('Unexpected fetch call');
        }

        return response;
    }) as typeof fetch;

    return {
        calls,
        restore() {
            globalThis.fetch = originalFetch;
        },
    };
}

test('returns configured dimensions without probing Ollama', async (t) => {
    const originalFetch = globalThis.fetch;
    let fetchCalls = 0;

    globalThis.fetch = (async () => {
        fetchCalls += 1;
        throw new Error('Fetch should not be called when dimensions are configured');
    }) as typeof fetch;

    t.after(() => {
        globalThis.fetch = originalFetch;
    });

    const embeddings = new OllamaEmbeddings({
        model: 'nomic-embed-text',
        baseUrl: 'http://localhost:11434',
        dimensions: 768,
    });

    await assert.doesNotReject(async () => {
        assert.equal(await embeddings.getDimensions(), 768);
    });
    assert.equal(fetchCalls, 0);
});

test('uses the current /api/embed response shape for dimension probing', async (t) => {
    const fetchStub = stubFetch([
        createJsonResponse({
            embeddings: [[0.1, 0.2, 0.3, 0.4]],
        }),
    ]);

    t.after(() => {
        fetchStub.restore();
    });

    const embeddings = new OllamaEmbeddings({
        model: 'nomic-embed-text',
        baseUrl: 'http://localhost:11434',
    });

    assert.equal(await embeddings.getDimensions(), 4);
    assert.equal(fetchStub.calls.length, 1);
    assert.equal(fetchStub.calls[0].url, 'http://localhost:11434/api/embed');
    assert.equal(fetchStub.calls[0].init?.method, 'POST');
    assert.deepEqual(JSON.parse(fetchStub.calls[0].init?.body as string), {
        model: 'nomic-embed-text',
        input: 'sample',
        truncate: false,
    });
});

test('falls back to legacy /api/embeddings when /api/embed fails', async (t) => {
    const fetchStub = stubFetch([
        createJsonResponse({ error: 'not found' }, { status: 404 }),
        createJsonResponse({ embedding: [1, 2, 3] }),
        createJsonResponse({ embedding: [4, 5, 6] }),
    ]);

    t.after(() => {
        fetchStub.restore();
    });

    const embeddings = new OllamaEmbeddings({
        model: 'nomic-embed-text',
        baseUrl: 'http://localhost:11434',
    });

    assert.deepEqual(await embeddings.embedDocuments(['first', 'second']), [
        [1, 2, 3],
        [4, 5, 6],
    ]);

    assert.equal(fetchStub.calls.length, 3);
    assert.equal(fetchStub.calls[0].url, 'http://localhost:11434/api/embed');
    assert.deepEqual(JSON.parse(fetchStub.calls[0].init?.body as string), {
        model: 'nomic-embed-text',
        input: ['first', 'second'],
        truncate: false,
    });
    assert.equal(fetchStub.calls[1].url, 'http://localhost:11434/api/embeddings');
    assert.deepEqual(JSON.parse(fetchStub.calls[1].init?.body as string), {
        model: 'nomic-embed-text',
        prompt: 'first',
    });
    assert.equal(fetchStub.calls[2].url, 'http://localhost:11434/api/embeddings');
    assert.deepEqual(JSON.parse(fetchStub.calls[2].init?.body as string), {
        model: 'nomic-embed-text',
        prompt: 'second',
    });
});