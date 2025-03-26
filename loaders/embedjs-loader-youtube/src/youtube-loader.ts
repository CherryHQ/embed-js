import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { YoutubeTranscript } from 'youtube-transcript';
import createDebugMessages from 'debug';
import md5 from 'md5';

import { BaseLoader } from '@cherrystudio/embedjs-interfaces';
import { cleanString } from '@cherrystudio/embedjs-utils';

export class YoutubeLoader extends BaseLoader<{ type: 'YoutubeLoader' }> {
    private readonly debug = createDebugMessages('embedjs:loader:YoutubeLoader');
    private readonly videoIdOrUrl: string;

    constructor({
        videoIdOrUrl,
        chunkSize,
        chunkOverlap,
    }: {
        videoIdOrUrl: string;
        chunkSize?: number;
        chunkOverlap?: number;
    }) {
        super(`YoutubeLoader_${md5(videoIdOrUrl)}`, { videoIdOrUrl }, chunkSize ?? 2000, chunkOverlap ?? 0);
        this.videoIdOrUrl = videoIdOrUrl;
    }

    override async *getUnfilteredChunks() {
        const chunker = new RecursiveCharacterTextSplitter({
            chunkSize: this.chunkSize,
            chunkOverlap: this.chunkOverlap,
        });

        try {
            const transcripts = await YoutubeTranscript.fetchTranscript(this.videoIdOrUrl, { lang: 'en' });
            this.debug(`Transcripts (length ${transcripts.length}) obtained for video`, this.videoIdOrUrl);

            for (const transcript of transcripts) {
                for (const chunk of await chunker.splitText(cleanString(transcript.text))) {
                    yield {
                        pageContent: chunk,
                        metadata: {
                            type: 'YoutubeLoader' as const,
                            source: this.videoIdOrUrl,
                        },
                    };
                }
            }
        } catch (e) {
            this.debug('Could not get transcripts for video', this.videoIdOrUrl, e);
        }
    }
}
