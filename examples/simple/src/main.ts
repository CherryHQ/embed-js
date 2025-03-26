import 'dotenv/config';
import { RAGApplicationBuilder, SIMPLE_MODELS } from '@cherrystudio/embedjs';
import { OpenAiEmbeddings } from '@cherrystudio/embedjs-openai';
import { WebLoader } from '@cherrystudio/embedjs-loader-web';
import { HNSWDb } from '@cherrystudio/embedjs-hnswlib';

const ragApplication = await new RAGApplicationBuilder()
    .setModel(SIMPLE_MODELS.OPENAI_GPT4_O)
    .setEmbeddingModel(new OpenAiEmbeddings())
    .setVectorDatabase(new HNSWDb())
    .build();

await ragApplication.addLoader(new WebLoader({ urlOrContent: 'https://www.forbes.com/profile/elon-musk' }));
await ragApplication.addLoader(new WebLoader({ urlOrContent: 'https://en.wikipedia.org/wiki/Elon_Musk' }));

await ragApplication.query('What is the net worth of Elon Musk today?');
