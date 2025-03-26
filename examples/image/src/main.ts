import 'dotenv/config';
import path from 'node:path';
import { RAGApplicationBuilder, SIMPLE_MODELS } from '@cherrystudio/embedjs';
import { ImageLoader } from '@cherrystudio/embedjs-loader-image';
import { OpenAiEmbeddings } from '@cherrystudio/embedjs-openai';
import { HNSWDb } from '@cherrystudio/embedjs-hnswlib';

const ragApplication = await new RAGApplicationBuilder()
    .setModel(SIMPLE_MODELS.OPENAI_GPT4_O)
    .setEmbeddingModel(new OpenAiEmbeddings())
    .setVectorDatabase(new HNSWDb())
    .build();

const imagePath = path.resolve('./examples/image/assets/test.jpg');
await ragApplication.addLoader(new ImageLoader({ filePathOrUrl: imagePath }));

await ragApplication.query('How does deep learning relate to artifical intelligence');
