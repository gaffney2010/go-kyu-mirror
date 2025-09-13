import javafx.application.Application
import javafx.application.Platform
import javafx.geometry.Pos
import javafx.scene.Scene
import javafx.scene.canvas.Canvas
import javafx.scene.control.Button
import javafx.scene.control.Label
import javafx.scene.layout.BorderPane
import javafx.scene.layout.HBox
import javafx.scene.paint.Color
import javafx.stage.Stage
import org.tensorflow.Result
import org.tensorflow.SavedModelBundle
import org.tensorflow.Tensor
import org.tensorflow.ndarray.NdArrays
import org.tensorflow.ndarray.Shape
import org.tensorflow.ndarray.buffer.DataBuffers
import org.tensorflow.types.TFloat32
import java.nio.FloatBuffer
import java.util.Optional
import kotlin.concurrent.thread

// Enum to represent the players or an empty intersection.
enum class Player {
    BLACK, WHITE, EMPTY;

    fun opponent(): Player {
        return when (this) {
            BLACK -> WHITE
            WHITE -> BLACK
            EMPTY -> EMPTY
        }
    }
}

// Data class to hold information about a single move.
data class Move(val player: Player, val x: Int, val y: Int)

/**
 * A simple SGF parser and game state container.
 * This class takes a raw SGF string, extracts the board size and the list of moves.
 * NOTE: This is a very basic parser for demonstration and only handles SZ, B, and W tags.
 */
class GoGame(sgf: String) {
    var size: Int = 19 // Default size
    val moves: List<Move>

    init {
        // Find board size, e.g., SZ[19]
        val sizeMatch = "SZ\\[(\\d+)\\]".toRegex().find(sgf)
        sizeMatch?.groupValues?.get(1)?.let {
            size = it.toInt()
        }

        // Find all moves, e.g., ;B[pd] or ;W[dp]
        moves = ";([BW])\\[([a-z]{2})\\]".toRegex().findAll(sgf).map { matchResult ->
            val (playerStr, posStr) = matchResult.destructured
            val player = if (playerStr == "B") Player.BLACK else Player.WHITE

            // SGF coordinates are 'a' through 's'. 'a' corresponds to index 0.
            val x = posStr[0] - 'a'
            val y = posStr[1] - 'a'

            Move(player, x, y)
        }.toList()
    }
}

/**
 * The main JavaFX Application class.
 * This sets up the UI (the window, canvas, buttons) and handles the core logic
 * for drawing the board and responding to user input.
 */
class GoBoardApp : Application() {

    // --- Configuration ---
    private val boardSize = 19 // We'll get the real size from SGF, but have a default.
    private val canvasSize = 800.0
    private val padding = 40.0
    private var cellWidth = (canvasSize - 2 * padding) / (boardSize - 1)

    // --- Game State ---
    private lateinit var game: GoGame
    private var currentMoveIndex = -1 // -1 represents the empty board before the first move.

    // Stores the board state at each point in the game's history.
    // This allows us to go backward without re-calculating the entire game.
    private val boardHistory = mutableListOf<Array<Array<Player>>>()

    // --- AI Model Properties ---
    private var model: SavedModelBundle? = null
    private var predictedMove: Pair<Int, Int>? = null // Store the predicted (x, y)

    // --- UI Elements ---
    private val canvas = Canvas(canvasSize, canvasSize)
    private val moveLabel = Label("Move: 0 / 0")

    override fun start(primaryStage: Stage) {
        // 1. Load the game data from a fixed file in the resources folder.
        val sgfData = try {
            val resourceStream = GoBoardApp::class.java.getResourceAsStream("/game.sgf")
            resourceStream!!.bufferedReader().readText()
        } catch (e: Exception) {
            println("ERROR: Could not load '/game.sgf'. Make sure it's in 'src/main/resources'.")
            println("Loading a default empty game as a fallback.")
            "(;FF[4]GM[1]SZ[19])"
        }

        game = GoGame(sgfData)
        // Update cellWidth in case the SGF had a different board size
        cellWidth = (canvasSize - 2 * padding) / (game.size - 1)

        // 2. Set up the initial board state (an empty board).
        val initialBoard = Array(game.size) { Array(game.size) { Player.EMPTY } }
        boardHistory.add(initialBoard)

        // 3. Configure UI controls (buttons).
        val prevButton = Button("< Prev").apply {
            setOnAction { navigateTo(currentMoveIndex - 1) }
        }
        val nextButton = Button("Next >").apply {
            setOnAction { navigateTo(currentMoveIndex + 1) }
        }

        // 4. Arrange UI elements in the window.
        val controls = HBox(20.0, prevButton, nextButton, moveLabel).apply {
            alignment = Pos.CENTER
            style = "-fx-padding: 10;"
        }

        val root = BorderPane().apply {
            center = canvas
            bottom = controls
            style = "-fx-background-color: #2b2b2b;"
        }

        // 5. Initial draw and update.
        updateMoveLabel()
        drawBoard()

        // Load the TensorFlow model
        try {
            // Get the absolute path from the system property we set in Gradle.
            val modelPath = System.getProperty("model.path", "exported_model")
            println("Attempting to load model from path: $modelPath") // For debugging

            model = SavedModelBundle.load(modelPath, "serve")
            println("TensorFlow model loaded successfully.")
        } catch (e: Exception) {
            println("ERROR: Could not load TensorFlow model from 'exported_model' directory.")
            e.printStackTrace()
        }

        // 6. Show the window.
        primaryStage.title = "Go Board Analyzer"
        primaryStage.scene = Scene(root)
        primaryStage.isResizable = false
        primaryStage.show()
    }

    // Clean up model resources on application exit
    override fun stop() {
        model?.close()
        super.stop()
    }

    /**
     * Main navigation function. Moves the game to the specified move number.
     */
    private fun navigateTo(newIndex: Int) {
        if (newIndex < -1 || newIndex >= game.moves.size) {
            return // Out of bounds
        }

        currentMoveIndex = newIndex

        if (currentMoveIndex >= 0 && currentMoveIndex >= boardHistory.size - 1) {
            val lastBoard = boardHistory.last().map { it.clone() }.toTypedArray()
            val move = game.moves[currentMoveIndex]

            // Place the new stone on the board
            lastBoard[move.y][move.x] = move.player

            // Check for and remove captured stones
            handleCaptures(lastBoard, move)

            // Add the new board state to our history.
            boardHistory.add(lastBoard)
        }

        updateMoveLabel()
        predictedMove = null // Clear old prediction
        drawBoard() // Redraw immediately to remove the old hint

        // Run prediction in the background
        runPrediction()
    }

    /**
     * After a move is played, checks adjacent opponent groups for captures.
     * Also checks for self-capture (suicide).
     */
    private fun handleCaptures(board: Array<Array<Player>>, move: Move) {
        val opponent = move.player.opponent()

        // Check neighbors for opponent groups to capture
        for (neighbor in getNeighbors(move.x, move.y)) {
            val (nx, ny) = neighbor
            if (board[ny][nx] == opponent) {
                val (liberties, group) = findGroup(board, nx, ny)
                if (liberties == 0) {
                    group.forEach { (gx, gy) -> board[gy][gx] = Player.EMPTY }
                }
            }
        }

        // Check for suicide move (only if no opponent stones were captured)
        val (myLiberties, myGroup) = findGroup(board, move.x, move.y)
        if (myLiberties == 0) {
            myGroup.forEach { (gx, gy) -> board[gy][gx] = Player.EMPTY }
        }
    }

    /**
     * Finds a group of connected stones and counts its liberties.
     * @return A Pair containing the number of liberties (Int) and a Set of stone coordinates (Pair<Int, Int>).
     */
    private fun findGroup(board: Array<Array<Player>>, startX: Int, startY: Int): Pair<Int, Set<Pair<Int, Int>>> {
        val player = board[startY][startX]
        if (player == Player.EMPTY) return 0 to emptySet()

        val queue = ArrayDeque<Pair<Int, Int>>()
        val visitedStones = mutableSetOf<Pair<Int, Int>>()
        val liberties = mutableSetOf<Pair<Int, Int>>()

        queue.add(startX to startY)
        visitedStones.add(startX to startY)

        while (queue.isNotEmpty()) {
            val (x, y) = queue.removeFirst()

            for (neighbor in getNeighbors(x, y)) {
                val (nx, ny) = neighbor
                val neighborState = board[ny][nx]

                if (neighborState == Player.EMPTY) {
                    liberties.add(neighbor)
                } else if (neighborState == player && neighbor !in visitedStones) {
                    visitedStones.add(neighbor)
                    queue.add(neighbor)
                }
            }
        }
        return liberties.size to visitedStones
    }

    /**
     * Gets valid neighbor coordinates for a given point.
     */
    private fun getNeighbors(x: Int, y: Int): List<Pair<Int, Int>> {
        val neighbors = mutableListOf<Pair<Int, Int>>()
        if (x > 0) neighbors.add(x - 1 to y)
        if (x < game.size - 1) neighbors.add(x + 1 to y)
        if (y > 0) neighbors.add(x to y - 1)
        if (y < game.size - 1) neighbors.add(x to y + 1)
        return neighbors
    }

    // --- AI Prediction Functions ---

    /**
     * Runs the loaded TensorFlow model to predict the next move.
     */
    private fun runPrediction() {
        // Guard clause: Don't predict if model isn't loaded or game is over.
        if (model == null || currentMoveIndex >= game.moves.size) {
            Platform.runLater {
                predictedMove = null
                drawBoard()
            }
            return
        }

        // Run inference on a background thread to prevent UI freezing
        thread(isDaemon = true) {
            try {
                val board = boardHistory[currentMoveIndex + 1]
                val nextPlayer = if (currentMoveIndex < 0) Player.BLACK else game.moves[currentMoveIndex].player.opponent()
                
                var finalPrediction: Pair<Int, Int>? = null

                // Create input tensor
                TFloat32.tensorOf(createInputShape(), DataBuffers.of(createInputBuffer(board, nextPlayer))).use { inputTensor ->

                    // Use the SavedModel signature instead of low-level Session API
                    val servingDefault = model!!.function("serving_default")
                    
                    // Create input map with the correct tensor name
                    val inputs = mapOf("keras_tensor" to inputTensor)
                    
                    // Run inference
                    val outputs = servingDefault.call(inputs)
                    
                    // Get the output tensor - extract from Optional
                    val outputTensorOpt = outputs["output_0"] as Optional<*>
                    val outputTensor = outputTensorOpt.get() as TFloat32

                    // Process Output Tensor - try direct access instead of copyTo
                    val probabilities = FloatArray(game.size * game.size + 1)

                    // Copy values manually instead of using copyTo
                    for (i in probabilities.indices) {
                        probabilities[i] = outputTensor.getFloat(0, i.toLong())
                    }

                    var bestMoveIndex = -1
                    var maxProb = -1.0f
                    for (i in probabilities.indices) {
                        val x = i % game.size
                        val y = i / game.size
                        val isPassMove = (i == game.size * game.size)
                        if (probabilities[i] > maxProb && (isPassMove || board[y][x] == Player.EMPTY)) {
                            maxProb = probabilities[i]
                            bestMoveIndex = i
                        }
                    }

                    finalPrediction = if (bestMoveIndex != -1 && bestMoveIndex < game.size * game.size) {
                        val x = bestMoveIndex % game.size
                        val y = bestMoveIndex / game.size
                        x to y
                    } else {
                        null
                    }
                }
                
                // Update the UI on the JavaFX Application Thread
                Platform.runLater {
                    predictedMove = finalPrediction
                    drawBoard()
                }

            } catch (e: Exception) {
                println("Error during model inference: ${e.message}")
                e.printStackTrace()
            }
        }
    }

    /** Helper function to define the tensor shape [1, 19, 19, 2] */
    private fun createInputShape(): Shape {
        return Shape.of(1, game.size.toLong(), game.size.toLong(), 2)
    }

    /** Helper function to convert our board array to the required FloatBuffer for the model */
    private fun createInputBuffer(board: Array<Array<Player>>, currentPlayer: Player): FloatBuffer {
<<<<<<< HEAD
        val opponent = currentPlayer.opponent()
=======
>>>>>>> b9b0be2314fe5fccf54cbd385883e471578ecd4a
        val buffer = FloatBuffer.allocate(1 * game.size * game.size * 2)
        for (y in 0 until game.size) {
            for (x in 0 until game.size) {
                when (board[y][x]) {
<<<<<<< HEAD
                    currentPlayer -> { buffer.put(1.0f); buffer.put(0.0f) }
                    opponent      -> { buffer.put(0.0f); buffer.put(1.0f) }
                    else          -> { buffer.put(0.0f); buffer.put(0.0f) }
=======
                    Player.BLACK -> { buffer.put(1.0f); buffer.put(0.0f) }  // Always black in channel 0
                    Player.WHITE -> { buffer.put(0.0f); buffer.put(1.0f) }  // Always white in channel 1
                    else         -> { buffer.put(0.0f); buffer.put(0.0f) }
>>>>>>> b9b0be2314fe5fccf54cbd385883e471578ecd4a
                }
            }
        }
        buffer.rewind()
        return buffer
    }

    private fun updateMoveLabel() {
        val displayMove = currentMoveIndex + 1
        moveLabel.text = "Move: $displayMove / ${game.moves.size}"
        moveLabel.style = "-fx-text-fill: #cccccc; -fx-font-size: 14px;"
    }

    private fun drawBoard() {
        val gc = canvas.graphicsContext2D
        gc.clearRect(0.0, 0.0, canvas.width, canvas.height)

        gc.fill = Color.web("#d1a36e")
        gc.fillRect(0.0, 0.0, canvas.width, canvas.height)

        gc.stroke = Color.BLACK
        gc.lineWidth = 1.0
        for (i in 0 until game.size) {
            val pos = padding + i * cellWidth
            gc.strokeLine(pos, padding, pos, canvasSize - padding)
            gc.strokeLine(padding, pos, canvasSize - padding, pos)
        }

        val hoshiCoords = if (game.size == 19) listOf(3, 9, 15) else listOf()
        gc.fill = Color.BLACK
        for (x in hoshiCoords) {
            for (y in hoshiCoords) {
                val hoshiX = padding + x * cellWidth
                val hoshiY = padding + y * cellWidth
                gc.fillOval(hoshiX - 4, hoshiY - 4, 8.0, 8.0)
            }
        }

        drawStones()
    }

    private fun drawStones() {
        val gc = canvas.graphicsContext2D
        val stoneRadius = cellWidth / 2.0 * 0.95

        val currentBoard = boardHistory[currentMoveIndex + 1]
        val lastMove = if (currentMoveIndex >= 0) game.moves[currentMoveIndex] else null

        for (y in 0 until game.size) {
            for (x in 0 until game.size) {
                val player = currentBoard[y][x]
                if (player != Player.EMPTY) {
                    val centerX = padding + x * cellWidth
                    val centerY = padding + y * cellWidth

                    gc.fill = if (player == Player.BLACK) Color.BLACK else Color.WHITE
                    gc.fillOval(centerX - stoneRadius, centerY - stoneRadius, stoneRadius * 2, stoneRadius * 2)

                    if (player == Player.WHITE) {
                        gc.stroke = Color.BLACK
                        gc.lineWidth = 0.5
                        gc.strokeOval(centerX - stoneRadius, centerY - stoneRadius, stoneRadius * 2, stoneRadius * 2)
                    }

                    if (lastMove != null && x == lastMove.x && y == lastMove.y) {
                        gc.fill = if (lastMove.player == Player.BLACK) Color.WHITE else Color.BLACK
                        val dotRadius = stoneRadius * 0.3
                        gc.fillOval(centerX - dotRadius, centerY - dotRadius, dotRadius * 2, dotRadius * 2)
                    }
                }
            }
        }

        // Draw the predicted move hint
        predictedMove?.let { (px, py) ->
            val centerX = padding + px * cellWidth
            val centerY = padding + py * cellWidth
            val radius = cellWidth / 3.0 // Make it smaller than a full stone

            gc.globalAlpha = 0.6 // Make it semi-transparent
            gc.fill = Color.LIMEGREEN
            gc.fillOval(centerX - radius, centerY - radius, radius * 2, radius * 2)
            gc.globalAlpha = 1.0 // Reset alpha for other drawing operations
        }
    }
}

fun main() {
    Application.launch(GoBoardApp::class.java)
}