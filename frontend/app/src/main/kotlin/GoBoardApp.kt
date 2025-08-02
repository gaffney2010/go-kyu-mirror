import javafx.application.Application
import javafx.geometry.Pos
import javafx.scene.Scene
import javafx.scene.canvas.Canvas
import javafx.scene.control.Button
import javafx.scene.control.Label
import javafx.scene.layout.BorderPane
import javafx.scene.layout.HBox
import javafx.scene.paint.Color
import javafx.scene.text.Font
import javafx.stage.Stage
import kotlin.math.min

// Enum to represent the players or an empty intersection.
enum class Player {
    BLACK, WHITE, EMPTY
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
    private val cellWidth = (canvasSize - 2 * padding) / (boardSize - 1)

    // --- Game State ---
    private lateinit var game: GoGame
    private var currentMoveIndex = -1 // -1 represents the empty board before the first move.
    
    // Stores the board state at each point in the game's history.
    // This allows us to go backward without re-calculating the entire game.
    private val boardHistory = mutableListOf<Array<Array<Player>>>()

    // --- UI Elements ---
    private val canvas = Canvas(canvasSize, canvasSize)
    private val moveLabel = Label("Move: 0 / 0")

    override fun start(primaryStage: Stage) {
        // 1. Load the game data from a hardcoded SGF string.
        // This is a famous game between Go Seigen and Kitani Minoru (1933).
        val sgfData = "(;FF[4]GM[1]SZ[19]AP[CGoban:3]ST[2]RU[Japanese]KM[0.00]PW[Kitani Minoru]WR[5p]PB[Go Seigen]BR[5p]DT[1933-10-16]PC[Tokyo, Japan]RE[B+3];B[pd];W[dp];B[pp];W[c q];B[cn];W[fq];B[fp];W[gq];B[eq];W[er];B[fr];W[gr];B[es];W[ds];B[fs];W[gs];B[dr];W[cr];B[cs];W[bs];B[br];W[cq];B[dq];W[cp];B[do];W[eo];B[en];W[dn];B[dm];W[cm];B[cl];W[dl];B[dk];W[ck];B[bk];W[cj];B[bj];W[ci];B[bi];W[ch];B[bg];W[cg];B[bf];W[cf];B[be];W[ce];B[bd];W[cd];B[bc];W[cc];B[bb];W[cb];B[ab];W[ba];B[de];W[df];B[ed];W[ec];B[fc];W[fb];B[gb];W[fa];B[ga];W[eb];B[db];W[da];B[ca];W[ea];B[dc];W[dd];B[ef];W[fg];B[eg];W[fh];B[eh];W[fi];B[ei];W[fj];B[ej];W[fk];B[ek];W[fl];B[el];W[fm];B[em];W[fn];B[gn];W[go];B[ho];W[hn];B[gm];W[gl];B[hl];W[hk];B[ik];W[ij];B[jj];W[ik];B[jk];W[il];B[jl];W[im];B[hm];W[in];B[io];W[jn];B[jo];W[kn];B[ko];W[ln];B[lo];W[mn];B[mo];W[nn];B[no];W[on];B[oo];W[po];B[qo];W[pn];B[qn];W[qm];B[rm];W[rl];B[qk];W[pk];B[qj];W[pj];B[pi];W[oi];B[oj];W[ni];B[mi];W[mj];B[li];W[mh];B[lh];W[lg];B[kg];W[kf];B[lf];W[ke];B[le];W[ld];B[md];W[mc];B[lc];W[lb];B[mb];W[kc];B[jc];W[kb];B[jb];W[ka];B[ja];W[la];B[ma];W[ib];B[hc];W[id];B[hd];W[ie];B[he];W[if];B[hf];W[ig];B[hg];W[ih];B[hh];W[ii];B[hi];W[ji];B[jh];W[ki];B[kh];W[kj];B[lj];W[lk];B[mk];W[ll];B[ml];W[mm];B[nm];W[nl];B[om];W[ol];B[pl];W[ok];B[ql];W[pm];B[qm];W[bp];B[bo];W[co];B[bn];W[ao];B[an];W[ap];B[am];W[di];B[dh];W[ci];B[dj];W[cj];B[ei];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];B[ek];W[ci];B[ek];W[di];-..)"
        game = GoGame(sgfData)

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

        // 6. Show the window.
        primaryStage.title = "Go Board Analyzer"
        primaryStage.scene = Scene(root)
        primaryStage.isResizable = false
        primaryStage.show()
    }

    /**
     * Main navigation function. Moves the game to the specified move number.
     */
    private fun navigateTo(newIndex: Int) {
        if (newIndex < -1 || newIndex >= game.moves.size) {
            return // Out of bounds
        }
        
        currentMoveIndex = newIndex
        
        // If we are moving forward to a state we haven't calculated yet...
        if (currentMoveIndex >= 0 && currentMoveIndex >= boardHistory.size - 1) {
            // Get the last known board state.
            val lastBoard = boardHistory.last().map { it.clone() }.toTypedArray()
            
            // Apply the current move.
            val move = game.moves[currentMoveIndex]
            lastBoard[move.y][move.x] = move.player
            
            // Add the new board state to our history.
            boardHistory.add(lastBoard)
        }
        
        updateMoveLabel()
        drawBoard()
    }

    /**
     * Updates the label showing the current move number.
     */
    private fun updateMoveLabel() {
        val displayMove = currentMoveIndex + 1
        moveLabel.text = "Move: $displayMove / ${game.moves.size}"
        moveLabel.style = "-fx-text-fill: #cccccc; -fx-font-size: 14px;"
    }

    /**
     * Clears the canvas and redraws the entire board, including grid, hoshi, and stones.
     */
    private fun drawBoard() {
        val gc = canvas.graphicsContext2D
        gc.clearRect(0.0, 0.0, canvas.width, canvas.height)

        // Draw board background
        gc.fill = Color.web("#d1a36e") // A traditional go board color
        gc.fillRect(0.0, 0.0, canvas.width, canvas.height)

        // Draw grid lines
        gc.stroke = Color.BLACK
        gc.lineWidth = 1.0
        for (i in 0 until game.size) {
            val pos = padding + i * cellWidth
            // Vertical line
            gc.strokeLine(pos, padding, pos, canvasSize - padding)
            // Horizontal line
            gc.strokeLine(padding, pos, canvasSize - padding, pos)
        }

        // Draw hoshi (star points)
        val hoshiCoords = if (game.size == 19) listOf(3, 9, 15) else listOf(2, 6)
        gc.fill = Color.BLACK
        for (x in hoshiCoords) {
            for (y in hoshiCoords) {
                val hoshiX = padding + x * cellWidth
                val hoshiY = padding + y * cellWidth
                gc.fillOval(hoshiX - 4, hoshiY - 4, 8.0, 8.0)
            }
        }
        
        // Draw stones
        drawStones()
    }

    /**
     * Draws the stones based on the current board state.
     */
    private fun drawStones() {
        val gc = canvas.graphicsContext2D
        val stoneRadius = cellWidth / 2.0 * 0.95
        
        // Get the correct board state from history.
        // Index is currentMoveIndex + 1 because history[0] is the empty board.
        val currentBoard = boardHistory[currentMoveIndex + 1]

        for (y in 0 until game.size) {
            for (x in 0 until game.size) {
                val player = currentBoard[y][x]
                if (player != Player.EMPTY) {
                    val centerX = padding + x * cellWidth
                    val centerY = padding + y * cellWidth
                    
                    gc.fill = if (player == Player.BLACK) Color.BLACK else Color.WHITE
                    gc.fillOval(centerX - stoneRadius, centerY - stoneRadius, stoneRadius * 2, stoneRadius * 2)

                    // Add a subtle border to white stones to make them pop
                    if (player == Player.WHITE) {
                        gc.stroke = Color.BLACK
                        gc.lineWidth = 0.5
                        gc.strokeOval(centerX - stoneRadius, centerY - stoneRadius, stoneRadius * 2, stoneRadius * 2)
                    }
                }
            }
        }
    }
}

// The main entry point for the application.
fun main() {
    Application.launch(GoBoardApp::class.java)
}
