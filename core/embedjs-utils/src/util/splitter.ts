import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters'

export interface ChunkWithOffsets {
    text: string;
    startOffset: number;
    endOffset: number;
}


export async function splitTextWithOffsets(textToSplit: string, chunkSize:number, chunkOverlap:number): Promise<ChunkWithOffsets[]> {
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: chunkSize,
      chunkOverlap: chunkOverlap,
    })
    const textChunks = await textSplitter.splitText(textToSplit)
    const chunksWithOffsets: ChunkWithOffsets[] = []
    let currentSearchOffset = 0
    for (const chunkText of textChunks) {
      const startOffset = textToSplit.indexOf(chunkText, currentSearchOffset)
      if (startOffset !== -1) {
        const endOffset = startOffset + chunkText.length
        chunksWithOffsets.push({ text: chunkText, startOffset, endOffset })
        currentSearchOffset = startOffset + 1
      }
    }
    return chunksWithOffsets
  }
