// Include minimal required headers from Knoodle
#include "Knoodle.hpp"
#include "src/PolyFold.hpp"

// Conditionally include Alexander if UMFPACK is available
#ifdef USE_UMFPACK
#include "submodules/Tensors/Accelerate.hpp"
#include "src/KnotInvariants/Alexander_UMFPACK.hpp"
#endif

#include "src/PrimeKnotLookupTable.hpp"

// Include our bridge header
#include "bindings.h"

#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <memory>

using namespace Knoodle;
using namespace Tensors;
using namespace Tools;

// Define the types we need
using Real = Real64;
using Int = Int64;
using LInt = Int;
using BReal = Real64;
using PD_T = PlanarDiagram2<Int>;        // New diagram type
using PDC_T = PlanarDiagramComplex<Int>; // New diagram complex type
using OldPD_T = PlanarDiagram<Int>;      // Legacy type, needed for Alexander bridge
using Complex = Complex64;

// Define Alexander polynomial calculator type (only if UMFPACK available)
#ifdef USE_UMFPACK
using Alexander_T = Alexander_UMFPACK<Complex, Int>;
#endif

// AlexanderResult method implementations
std::string AlexanderResult::to_string() const {
    if (exponent == 0) {
        return std::to_string(mantissa);
    } else {
        std::ostringstream ss;
        ss << mantissa << "e" << exponent;
        return ss.str();
    }
}

double AlexanderResult::to_double() const {
    return mantissa * std::pow(10.0, exponent);
}

// Implementation class that holds the planar diagram complex
class KnotAnalyzerImpl {
public:
    PDC_T pdc;                // The diagram complex (replaces old unique_ptr<PD_T>)
    Int primary_diagram_idx;  // Index of the primary (first non-unknot) diagram, -1 if none

#ifdef USE_UMFPACK
    mutable std::unique_ptr<Alexander_T> alexander_calc;
#endif

    KnotAnalyzerImpl() : primary_diagram_idx(-1) {}

    KnotAnalyzerImpl(const KnotAnalyzerImpl& other)
        : pdc(other.pdc), primary_diagram_idx(other.primary_diagram_idx) {
        // Don't copy alexander_calc - it will be created on-demand
    }

#ifdef USE_UMFPACK
    KnotAnalyzerImpl(KnotAnalyzerImpl&& other) noexcept
        : pdc(std::move(other.pdc)),
          primary_diagram_idx(other.primary_diagram_idx),
          alexander_calc(std::move(other.alexander_calc)) {}
#else
    KnotAnalyzerImpl(KnotAnalyzerImpl&& other) noexcept
        : pdc(std::move(other.pdc)),
          primary_diagram_idx(other.primary_diagram_idx) {}
#endif

    // Get the primary diagram (first non-unknot with crossings)
    const PD_T* get_primary_diagram() const {
        if (primary_diagram_idx >= 0 && primary_diagram_idx < pdc.DiagramCount()) {
            return &pdc.Diagram(primary_diagram_idx);
        }
        return nullptr;
    }

#ifdef USE_UMFPACK
    // Get or create Alexander calculator
    Alexander_T& get_alexander_calculator() const {
        if (!alexander_calc) {
            alexander_calc = std::make_unique<Alexander_T>();
        }
        return *alexander_calc;
    }
#endif
};

// Calculate squared gyradius from coordinates
double calculate_squared_gyradius(const std::vector<double>& coords) {
    int n_points = coords.size() / 3;
    if (n_points == 0) return 0.0;

    // Calculate center of mass
    double x_sum = 0.0, y_sum = 0.0, z_sum = 0.0;
    for (int i = 0; i < n_points; ++i) {
        x_sum += coords[3*i];
        y_sum += coords[3*i+1];
        z_sum += coords[3*i+2];
    }
    double x_avg = x_sum / n_points;
    double y_avg = y_sum / n_points;
    double z_avg = z_sum / n_points;

    // Calculate average squared distance from center
    double sum_sq_dist = 0.0;
    for (int i = 0; i < n_points; ++i) {
        double dx = coords[3*i] - x_avg;
        double dy = coords[3*i+1] - y_avg;
        double dz = coords[3*i+2] - z_avg;
        sum_sq_dist += dx*dx + dy*dy + dz*dz;
    }

    return sum_sq_dist / n_points;
}

// Convert PD_T (PlanarDiagram2) to PD code string (5 entries: 4 arcs + handedness)
std::string pd_to_string(const PD_T& pd) {
    std::stringstream ss;
    auto pdcode = pd.PDCode();

    for (Int i = 0; i < pdcode.Dimension(0); ++i) {
        ss << "[";
        for (Int j = 0; j < 5; ++j) {  // Include all 5 entries (4 arcs + handedness)
            ss << pdcode(i, j);
            if (j < 4) ss << ",";  // Comma after first 4 entries
        }
        ss << "]";
        if (i < pdcode.Dimension(0) - 1) ss << ",";
    }

    return ss.str();
}

// Convert PD_T to unsigned PD code string (4 entries only, for backward compatibility)
std::string pd_to_string_unsigned(const PD_T& pd) {
    std::stringstream ss;
    auto pdcode = pd.PDCode();

    for (Int i = 0; i < pdcode.Dimension(0); ++i) {
        ss << "[";
        for (Int j = 0; j < 4; ++j) {  // Only first 4 entries (arcs only)
            ss << pdcode(i, j);
            if (j < 3) ss << ",";
        }
        ss << "]";
        if (i < pdcode.Dimension(0) - 1) ss << ",";
    }

    return ss.str();
}

// Convert PD_T to Gauss code string
std::string gauss_to_string(const PD_T& pd) {
    try {
        auto extgausscode = pd.ExtendedGaussCode();
        Int code_size = extgausscode.Size();

        if (code_size == 0) {
            return "";
        }

        std::stringstream ss;
        ss << "ext:";
        for (Int i = 0; i < code_size; ++i) {
            ss << extgausscode[i];
            if (i < code_size - 1) ss << " ";
        }

        return ss.str();
    } catch (...) {
        return "";
    }
}

// Map old simplify levels to new Simplify_Args_T
PDC_T::Simplify_Args_T make_simplify_args(int simplify_level) {
    PDC_T::Simplify_Args_T args;
    switch (simplify_level) {
        case 1:
        case 2:
        case 3:
            // Strand simplification only - no disconnect, split, or reapr
            args.disconnectQ = false;
            args.splitQ = false;
            args.reapr_embedding_trials = 0;
            args.reapr_rotation_trials = 0;
            break;
        case 4:
            // Add disconnect and split, but no reapr
            args.disconnectQ = true;
            args.splitQ = true;
            args.reapr_embedding_trials = 0;
            args.reapr_rotation_trials = 0;
            break;
        case 5:
        default:
            // Full simplification with all defaults (including reapr)
            break;
    }
    return args;
}

// Analyze a PDC_T to extract summary properties
struct PDCAnalysis {
    Int total_crossing_count;
    Int total_writhe;
    Int unlink_count;
    Int nontrivial_count;
    Int first_nontrivial_idx;

    static PDCAnalysis analyze(const PDC_T& pdc) {
        PDCAnalysis result = {};
        result.total_crossing_count = 0;
        result.total_writhe = 0;
        result.unlink_count = 0;
        result.nontrivial_count = 0;
        result.first_nontrivial_idx = -1;

        for (Int i = 0; i < pdc.DiagramCount(); ++i) {
            const PD_T& pd = pdc.Diagram(i);
            if (pd.ProvenUnknotQ()) {
                result.unlink_count++;
            } else if (pd.CrossingCount() > 0) {
                result.total_crossing_count += pd.CrossingCount();
                result.total_writhe += pd.Writhe();
                result.nontrivial_count++;
                if (result.first_nontrivial_idx < 0) {
                    result.first_nontrivial_idx = i;
                }
            }
        }

        return result;
    }
};

// Bridge: construct an old PlanarDiagram from a new PlanarDiagram2's PD code
// Needed because Alexander_UMFPACK still uses PlanarDiagram<Int>
OldPD_T bridge_to_old_pd(const PD_T& pd2) {
    auto pdcode = pd2.PDCode();
    Int n = pd2.CrossingCount();
    if (n == 0) {
        return OldPD_T();
    }
    return OldPD_T::FromSignedPDCode(pdcode.data(), n, Int(0), true, false);
}

// KnotAnalyzer implementation
KnotAnalyzer::KnotAnalyzer() : impl(std::make_shared<KnotAnalyzerImpl>()) {
    crossing_count = 0;
    writhe = 0;
    squared_gyradius = 0.0;
    link_component_count = 1;
    unlink_count = 0;
    is_prime = true;
    is_composite = false;
    prime_component_count = 0;
}

KnotAnalyzer::KnotAnalyzer(const std::vector<double>& coordinates, bool simplify, int simplify_level)
    : impl(std::make_shared<KnotAnalyzerImpl>()) {

    try {
        Int n = coordinates.size() / 3;

        // Create planar diagram complex from coordinates
        impl->pdc = PDC_T::FromKnotEmbedding(coordinates.data(), n);

        // Apply simplification if requested
        if (simplify) {
            auto args = make_simplify_args(simplify_level);
            impl->pdc.Simplify(args);
        }

        // Analyze the complex
        auto analysis = PDCAnalysis::analyze(impl->pdc);
        impl->primary_diagram_idx = analysis.first_nontrivial_idx;

        // Extract properties
        crossing_count = analysis.total_crossing_count;
        writhe = analysis.total_writhe;
        unlink_count = analysis.unlink_count;
        squared_gyradius = calculate_squared_gyradius(coordinates);

        // PD code and gauss code from the primary diagram
        const PD_T* primary = impl->get_primary_diagram();
        if (primary) {
            pd_code = pd_to_string(*primary);
            gauss_code = gauss_to_string(*primary);
            link_component_count = primary->LinkComponentCount();
        } else {
            pd_code = "";
            gauss_code = "";
            link_component_count = 1;  // unknot has 1 link component
        }

        // Prime decomposition: every non-unknot diagram is a prime component
        if (analysis.nontrivial_count > 1) {
            is_composite = true;
            is_prime = false;
            prime_component_count = analysis.nontrivial_count;

            for (Int i = 0; i < impl->pdc.DiagramCount(); ++i) {
                const PD_T& comp_pd = impl->pdc.Diagram(i);
                if (comp_pd.ProvenUnknotQ() || comp_pd.CrossingCount() == 0) continue;

                KnotAnalyzer comp;
                comp.crossing_count = comp_pd.CrossingCount();
                comp.writhe = comp_pd.Writhe();
                comp.pd_code = pd_to_string(comp_pd);
                comp.gauss_code = gauss_to_string(comp_pd);
                comp.squared_gyradius = 0.0;
                comp.link_component_count = comp_pd.LinkComponentCount();
                comp.unlink_count = 0;
                comp.is_prime = true;
                comp.is_composite = false;
                comp.prime_component_count = 1;

                // Create a PDC_T for this child from a copy of the component diagram
                comp.impl = std::make_shared<KnotAnalyzerImpl>();
                PD_T comp_copy = comp_pd;
                comp.impl->pdc = PDC_T(std::move(comp_copy));
                comp.impl->primary_diagram_idx = 0;

                prime_components.push_back(std::move(comp));
            }
        } else if (analysis.nontrivial_count == 1) {
            is_prime = true;
            is_composite = false;
            prime_component_count = 1;
        } else {
            // Unknot
            is_prime = true;
            is_composite = false;
            prime_component_count = 0;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error in KnotAnalyzer constructor: " << e.what() << std::endl;
        // Set error values
        crossing_count = -1;
        writhe = 0;
        pd_code = "Error: " + std::string(e.what());
        gauss_code = "Error: " + std::string(e.what());
        squared_gyradius = 0.0;
        link_component_count = -1;
        unlink_count = -1;
        is_prime = false;
        is_composite = false;
        prime_component_count = -1;
    }
}

// Copy constructor
KnotAnalyzer::KnotAnalyzer(const KnotAnalyzer& other)
    : impl(std::make_shared<KnotAnalyzerImpl>(*other.impl)),
      crossing_count(other.crossing_count),
      writhe(other.writhe),
      pd_code(other.pd_code),
      gauss_code(other.gauss_code),
      squared_gyradius(other.squared_gyradius),
      link_component_count(other.link_component_count),
      unlink_count(other.unlink_count),
      is_prime(other.is_prime),
      is_composite(other.is_composite),
      prime_component_count(other.prime_component_count),
      prime_components(other.prime_components) {
}

// Move constructor
KnotAnalyzer::KnotAnalyzer(KnotAnalyzer&& other) noexcept
    : impl(std::move(other.impl)),
      crossing_count(other.crossing_count),
      writhe(other.writhe),
      pd_code(std::move(other.pd_code)),
      gauss_code(std::move(other.gauss_code)),
      squared_gyradius(other.squared_gyradius),
      link_component_count(other.link_component_count),
      unlink_count(other.unlink_count),
      is_prime(other.is_prime),
      is_composite(other.is_composite),
      prime_component_count(other.prime_component_count),
      prime_components(std::move(other.prime_components)) {
}

// Copy assignment
KnotAnalyzer& KnotAnalyzer::operator=(const KnotAnalyzer& other) {
    if (this != &other) {
        impl = std::make_shared<KnotAnalyzerImpl>(*other.impl);
        crossing_count = other.crossing_count;
        writhe = other.writhe;
        pd_code = other.pd_code;
        gauss_code = other.gauss_code;
        squared_gyradius = other.squared_gyradius;
        link_component_count = other.link_component_count;
        unlink_count = other.unlink_count;
        is_prime = other.is_prime;
        is_composite = other.is_composite;
        prime_component_count = other.prime_component_count;
        prime_components = other.prime_components;
    }
    return *this;
}

// Move assignment
KnotAnalyzer& KnotAnalyzer::operator=(KnotAnalyzer&& other) noexcept {
    if (this != &other) {
        impl = std::move(other.impl);
        crossing_count = other.crossing_count;
        writhe = other.writhe;
        pd_code = std::move(other.pd_code);
        gauss_code = std::move(other.gauss_code);
        squared_gyradius = other.squared_gyradius;
        link_component_count = other.link_component_count;
        unlink_count = other.unlink_count;
        is_prime = other.is_prime;
        is_composite = other.is_composite;
        prime_component_count = other.prime_component_count;
        prime_components = std::move(other.prime_components);
    }
    return *this;
}

// Destructor
KnotAnalyzer::~KnotAnalyzer() = default;

// Helper methods for PD code analysis
std::string KnotAnalyzer::get_pd_code_unsigned() const {
    const PD_T* primary = impl->get_primary_diagram();
    if (!primary) return "";
    return pd_to_string_unsigned(*primary);
}

std::vector<std::vector<int>> KnotAnalyzer::get_pd_code_matrix() const {
    std::vector<std::vector<int>> result;

    const PD_T* primary = impl->get_primary_diagram();
    if (!primary) return result;

    auto pdcode = primary->PDCode();
    int num_crossings = pdcode.Dimension(0);

    result.reserve(num_crossings);

    for (int i = 0; i < num_crossings; ++i) {
        std::vector<int> crossing(5);  // 4 arcs + handedness
        for (int j = 0; j < 5; ++j) {
            crossing[j] = static_cast<int>(pdcode(i, j));
        }
        result.push_back(crossing);
    }

    return result;
}

std::vector<int> KnotAnalyzer::get_crossing_handedness() const {
    std::vector<int> handedness;

    const PD_T* primary = impl->get_primary_diagram();
    if (!primary) return handedness;

    auto pdcode = primary->PDCode();
    int num_crossings = pdcode.Dimension(0);

    handedness.reserve(num_crossings);

    for (int i = 0; i < num_crossings; ++i) {
        handedness.push_back(static_cast<int>(pdcode(i, 4)));  // 5th entry is handedness
    }

    return handedness;
}

// Alexander polynomial methods
#ifdef USE_UMFPACK
AlexanderResult KnotAnalyzer::alexander(const std::complex<double>& z) const {
    AlexanderResult result;

    const PD_T* primary = impl->get_primary_diagram();
    if (!primary || primary->CrossingCount() == 0) {
        result.mantissa = 1.0;
        result.exponent = 0;
        return result;
    }

    try {
        // Bridge to old PlanarDiagram for Alexander computation
        OldPD_T old_pd = bridge_to_old_pd(*primary);

        Alexander_T& alex_calc = impl->get_alexander_calculator();

        Complex arg = Complex(z.real(), z.imag());
        Complex mantissa;
        Int exponent;

        // Use batch API (single element) because the single-value Alexander()
        // takes mantissa/exponent by value, so results are never written back.
        alex_calc.Alexander(old_pd, &arg, Int(1), &mantissa, &exponent, false);

        // Convert complex result to real (Alexander polynomial should be real for real inputs)
        result.mantissa = std::real(mantissa);
        result.exponent = exponent;

    } catch (const std::exception& e) {
        std::cerr << "Error computing Alexander polynomial: " << e.what() << std::endl;
        result.mantissa = 0.0;
        result.exponent = 0;
    }

    return result;
}

AlexanderResult KnotAnalyzer::alexander(double t) const {
    return alexander(std::complex<double>(t, 0.0));
}

std::vector<AlexanderResult> KnotAnalyzer::alexander(const std::vector<std::complex<double>>& points) const {
    std::vector<AlexanderResult> results;
    results.reserve(points.size());

    const PD_T* primary = impl->get_primary_diagram();
    if (!primary || primary->CrossingCount() == 0) {
        // Unknot case
        for (size_t i = 0; i < points.size(); ++i) {
            results.push_back({1.0, 0});
        }
        return results;
    }

    try {
        // Bridge to old PlanarDiagram for Alexander computation
        OldPD_T old_pd = bridge_to_old_pd(*primary);

        Alexander_T& alex_calc = impl->get_alexander_calculator();

        // Convert input points to Complex
        std::vector<Complex> args;
        args.reserve(points.size());
        for (const auto& p : points) {
            args.emplace_back(p.real(), p.imag());
        }

        // Prepare output arrays
        std::vector<Complex> mantissas(points.size());
        std::vector<Int> exponents(points.size());

        // Compute Alexander polynomial for all points
        alex_calc.Alexander(old_pd, args.data(), static_cast<Int>(args.size()),
                           mantissas.data(), exponents.data(), false);

        // Convert results
        for (size_t i = 0; i < points.size(); ++i) {
            AlexanderResult result;
            result.mantissa = std::real(mantissas[i]);
            result.exponent = exponents[i];
            results.push_back(result);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error computing Alexander polynomial batch: " << e.what() << std::endl;
        // Return zeros on error
        for (size_t i = 0; i < points.size(); ++i) {
            results.push_back({0.0, 0});
        }
    }

    return results;
}

std::vector<AlexanderResult> KnotAnalyzer::alexander(const std::vector<double>& points) const {
    std::vector<std::complex<double>> complex_points;
    complex_points.reserve(points.size());
    for (double t : points) {
        complex_points.emplace_back(t, 0.0);
    }

    return alexander(complex_points);
}

std::map<std::string, AlexanderResult> KnotAnalyzer::alexander_invariants() const {
    std::map<std::string, AlexanderResult> invariants;

    // Common evaluation points for Alexander polynomial
    std::vector<std::pair<std::string, std::complex<double>>> test_points = {
        {"at_minus_1", std::complex<double>(-1.0, 0.0)},
        {"at_1", std::complex<double>(1.0, 0.0)},
        {"at_i", std::complex<double>(0.0, 1.0)},
        {"at_minus_i", std::complex<double>(0.0, -1.0)},
        {"at_omega", std::complex<double>(-0.5, std::sqrt(3.0)/2.0)}, // primitive cube root of unity
        {"at_omega2", std::complex<double>(-0.5, -std::sqrt(3.0)/2.0)} // omega^2
    };

    for (const auto& [name, point] : test_points) {
        invariants[name] = alexander(point);
    }

    return invariants;
}
#else
// Stub implementations when UMFPACK not available
AlexanderResult KnotAnalyzer::alexander(const std::complex<double>&) const {
    std::cerr << "Alexander polynomial requires UMFPACK (not available)" << std::endl;
    return {0.0, 0};
}

AlexanderResult KnotAnalyzer::alexander(double) const {
    std::cerr << "Alexander polynomial requires UMFPACK (not available)" << std::endl;
    return {0.0, 0};
}

std::vector<AlexanderResult> KnotAnalyzer::alexander(const std::vector<std::complex<double>>& points) const {
    std::cerr << "Alexander polynomial requires UMFPACK (not available)" << std::endl;
    return std::vector<AlexanderResult>(points.size(), {0.0, 0});
}

std::vector<AlexanderResult> KnotAnalyzer::alexander(const std::vector<double>& points) const {
    std::cerr << "Alexander polynomial requires UMFPACK (not available)" << std::endl;
    return std::vector<AlexanderResult>(points.size(), {0.0, 0});
}

std::map<std::string, AlexanderResult> KnotAnalyzer::alexander_invariants() const {
    std::cerr << "Alexander polynomial requires UMFPACK (not available)" << std::endl;
    return {};
}
#endif

// Convenience functions that create a KnotAnalyzer internally
std::string get_pd_code(const std::vector<double>& coordinates, bool simplify) {
    KnotAnalyzer analyzer(coordinates, simplify);
    return analyzer.get_pd_code();
}

std::string get_pd_code_unsigned(const std::vector<double>& coordinates, bool simplify) {
    KnotAnalyzer analyzer(coordinates, simplify);
    return analyzer.get_pd_code_unsigned();
}

std::string get_gauss_code(const std::vector<double>& coordinates, bool simplify) {
    KnotAnalyzer analyzer(coordinates, simplify);
    return analyzer.get_gauss_code();
}

bool is_unknot(const std::vector<double>& coordinates) {
    KnotAnalyzer analyzer(coordinates, true);
    return analyzer.is_unknot();
}

AlexanderResult alexander(const std::vector<double>& coordinates, double t, bool simplify) {
    KnotAnalyzer analyzer(coordinates, simplify);
    return analyzer.alexander(t);
}

AlexanderResult alexander(const std::vector<double>& coordinates, const std::complex<double>& z, bool simplify) {
    KnotAnalyzer analyzer(coordinates, simplify);
    return analyzer.alexander(z);
}

// MacLeod code methods
std::vector<uint8_t> KnotAnalyzer::macleod_code() const {
    std::vector<uint8_t> result;

    const PD_T* primary = impl->get_primary_diagram();
    if (!primary || primary->CrossingCount() == 0) {
        return result;  // Empty for unknot
    }

    if (primary->LinkComponentCount() > 1) {
        std::cerr << "MacLeod code not defined for links with multiple components" << std::endl;
        return result;
    }

    try {
        auto code = primary->template MacLeodCode<UInt8>();
        result.reserve(code.Size());
        for (Int i = 0; i < code.Size(); ++i) {
            result.push_back(code[i]);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error computing MacLeod code: " << e.what() << std::endl;
    }

    return result;
}

std::string KnotAnalyzer::macleod_string() const {
    const PD_T* primary = impl->get_primary_diagram();
    if (!primary || primary->CrossingCount() == 0) {
        return "";
    }

    if (primary->LinkComponentCount() > 1) {
        return "";
    }

    try {
        return primary->MacLeodString();
    } catch (const std::exception& e) {
        std::cerr << "Error computing MacLeod string: " << e.what() << std::endl;
        return "";
    }
}

bool KnotAnalyzer::proven_minimal() const {
    const PD_T* primary = impl->get_primary_diagram();
    if (!primary) return false;
    return primary->ProvenMinimalQ();
}

// Factory method to create from PD code
KnotAnalyzer KnotAnalyzer::from_pd_code(const std::vector<std::vector<int>>& pd_code, bool simplify, int simplify_level) {
    KnotAnalyzer result;

    if (pd_code.empty()) {
        // Empty PD code = unknot
        result.crossing_count = 0;
        result.writhe = 0;
        result.link_component_count = 1;
        result.unlink_count = 0;
        result.is_prime = true;
        result.is_composite = false;
        result.prime_component_count = 0;
        return result;
    }

    try {
        Int n = static_cast<Int>(pd_code.size());  // Number of crossings

        // Determine if PD code has 4 or 5 entries per crossing
        bool signed_pd = (pd_code[0].size() == 5);

        // Flatten PD code
        std::vector<Int> flat_pd;
        int entries_per_crossing = signed_pd ? 5 : 4;
        flat_pd.reserve(n * entries_per_crossing);

        for (const auto& crossing : pd_code) {
            if (static_cast<int>(crossing.size()) != entries_per_crossing) {
                throw std::runtime_error("Inconsistent PD code entry sizes");
            }
            for (int val : crossing) {
                flat_pd.push_back(static_cast<Int>(val));
            }
        }

        // Create PlanarDiagramComplex from PD code
        result.impl = std::make_shared<KnotAnalyzerImpl>();
        if (signed_pd) {
            result.impl->pdc = PDC_T::FromPDCode<true>(flat_pd.data(), n, false, false);
        } else {
            result.impl->pdc = PDC_T::FromPDCode<false>(flat_pd.data(), n, false, false);
        }

        // Validate
        if (result.impl->pdc.DiagramCount() == 0 ||
            result.impl->pdc.Diagram(0).InvalidQ()) {
            throw std::runtime_error("Invalid PD code - could not create valid diagram");
        }

        // Apply simplification if requested
        if (simplify) {
            auto args = make_simplify_args(simplify_level);
            result.impl->pdc.Simplify(args);
        }

        // Analyze the complex
        auto analysis = PDCAnalysis::analyze(result.impl->pdc);
        result.impl->primary_diagram_idx = analysis.first_nontrivial_idx;

        // Extract properties
        result.crossing_count = analysis.total_crossing_count;
        result.writhe = analysis.total_writhe;
        result.unlink_count = analysis.unlink_count;
        result.squared_gyradius = 0.0;  // No coordinates available

        const PD_T* primary = result.impl->get_primary_diagram();
        if (primary) {
            result.pd_code = pd_to_string(*primary);
            result.gauss_code = gauss_to_string(*primary);
            result.link_component_count = primary->LinkComponentCount();
        } else {
            result.pd_code = "";
            result.gauss_code = "";
            result.link_component_count = 1;
        }

        // Prime decomposition
        if (analysis.nontrivial_count > 1) {
            result.is_composite = true;
            result.is_prime = false;
            result.prime_component_count = analysis.nontrivial_count;

            for (Int i = 0; i < result.impl->pdc.DiagramCount(); ++i) {
                const PD_T& comp_pd = result.impl->pdc.Diagram(i);
                if (comp_pd.ProvenUnknotQ() || comp_pd.CrossingCount() == 0) continue;

                KnotAnalyzer comp;
                comp.impl = std::make_shared<KnotAnalyzerImpl>();
                PD_T comp_copy = comp_pd;
                comp.impl->pdc = PDC_T(std::move(comp_copy));
                comp.impl->primary_diagram_idx = 0;
                comp.crossing_count = comp_pd.CrossingCount();
                comp.writhe = comp_pd.Writhe();
                comp.pd_code = pd_to_string(comp_pd);
                comp.gauss_code = gauss_to_string(comp_pd);
                comp.squared_gyradius = 0.0;
                comp.link_component_count = comp_pd.LinkComponentCount();
                comp.unlink_count = 0;
                comp.is_prime = true;
                comp.is_composite = false;
                comp.prime_component_count = 1;

                result.prime_components.push_back(std::move(comp));
            }
        } else if (analysis.nontrivial_count == 1) {
            result.is_prime = true;
            result.is_composite = false;
            result.prime_component_count = 1;
        } else {
            result.is_prime = true;
            result.is_composite = false;
            result.prime_component_count = 0;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error creating KnotAnalyzer from PD code: " << e.what() << std::endl;
        result.crossing_count = -1;
        result.pd_code = "Error: " + std::string(e.what());
    }

    return result;
}
