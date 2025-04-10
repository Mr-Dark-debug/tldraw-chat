import type { NextApiRequest, NextApiResponse } from 'next'

type DiagramRequest = {
  prompt: string
}

type DiagramResponse = {
  changes: any[]
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse<DiagramResponse | { error: string }>
) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  try {
    const { prompt } = req.body as DiagramRequest

    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required' })
    }

    console.log('Generating diagram for prompt:', prompt)

    // Forward the request to the backend Python service
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'
    const response = await fetch(`${backendUrl}/api/generate-diagram`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt }),
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error('Backend error:', errorText)
      return res.status(response.status).json({ error: `Backend error: ${errorText}` })
    }

    const data = await response.json()
    
    // Ensure we're handling the response correctly
    console.log('Backend response:', data)
    
    // The backend should return a nested changes array
    // Make sure we're returning a properly formatted array
    if (data && data.changes && Array.isArray(data.changes)) {
      return res.status(200).json({ changes: data.changes })
    } else if (Array.isArray(data)) {
      // In case the backend returns the changes directly as an array
      return res.status(200).json({ changes: data })
    } else {
      console.error('Invalid response format from backend:', data)
      return res.status(500).json({ error: 'Invalid response format from backend' })
    }
  } catch (error) {
    console.error('Error generating diagram:', error)
    return res.status(500).json({ error: 'Failed to generate diagram' })
  }
} 