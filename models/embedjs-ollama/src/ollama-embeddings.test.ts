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

test('returns configured dimensions without probing Ollama', async () => {
    const originalFetch = globalThis.fetch;
    let fetchCalls = 0;

    globalThis.fetch = (async () => {
        fetchCalls += 1;
        throw new Error('Fetch should not be called when dimensions are configured');
    }) as typeof fetch;

    try {
        const embeddings = new OllamaEmbeddings({
            model: 'nomic-embed-text',
            baseUrl: 'http://localhost:11434',
            dimensions: 768,
        });

        await assert.doesNotReject(async () => {
            assert.equal(await embeddings.getDimensions(), 768);
        });
        assert.equal(fetchCalls, 0);
    } finally {
        globalThis.fetch = originalFetch;
    }
});

test('uses the current /api/embed response shape for dimension probing', async () => {
    const fetchStub = stubFetch([
        createJsonResponse({
            embeddings: [[0.1, 0.2, 0.3, 0.4]],
        }),
    ]);

    try {
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
        });
    } finally {
        fetchStub.restore();
    }
});

test('preserves an explicitly configured truncate flag', async () => {
    const fetchStub = stubFetch([
        createJsonResponse({
            embeddings: [[0.1, 0.2, 0.3, 0.4]],
        }),
    ]);

    try {
        const embeddings = new OllamaEmbeddings({
            model: 'nomic-embed-text',
            baseUrl: 'http://localhost:11434',
            truncate: true,
        });

        assert.equal(await embeddings.getDimensions(), 4);
        assert.equal(fetchStub.calls.length, 1);
        assert.deepEqual(JSON.parse(fetchStub.calls[0].init?.body as string), {
            model: 'nomic-embed-text',
            input: 'sample',
            truncate: true,
        });
    } finally {
        fetchStub.restore();
    }
});

test('falls back to legacy /api/embeddings when /api/embed fails', async () => {
    const fetchStub = stubFetch([
        createJsonResponse({ error: 'not found' }, { status: 404 }),
        createJsonResponse({ embedding: [1, 2, 3] }),
        createJsonResponse({ embedding: [4, 5, 6] }),
    ]);

    try {
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
    } finally {
        fetchStub.restore();
    }
});

test('does not fall back to legacy /api/embeddings for non-compatibility failures', async () => {
    const fetchStub = stubFetch([new Response('upstream failure', { status: 500 })]);

    try {
        const embeddings = new OllamaEmbeddings({
            model: 'nomic-embed-text',
            baseUrl: 'http://localhost:11434',
        });

        await assert.rejects(async () => embeddings.embedDocuments(['first']), /upstream failure/);
        assert.equal(fetchStub.calls.length, 1);
        assert.equal(fetchStub.calls[0].url, 'http://localhost:11434/api/embed');
    } finally {
        fetchStub.restore();
    }
});

test('does not fall back to legacy /api/embeddings when dimensions are configured', async () => {
    const fetchStub = stubFetch([new Response('missing current api', { status: 404 })]);

    try {
        const embeddings = new OllamaEmbeddings({
            model: 'nomic-embed-text',
            baseUrl: 'http://localhost:11434',
            dimensions: 768,
        });

        await assert.rejects(async () => embeddings.embedDocuments(['first']), /missing current api/);
        assert.equal(fetchStub.calls.length, 1);
        assert.deepEqual(JSON.parse(fetchStub.calls[0].init?.body as string), {
            model: 'nomic-embed-text',
            input: 'first',
            dimensions: 768,
        });
    } finally {
        fetchStub.restore();
    }
});